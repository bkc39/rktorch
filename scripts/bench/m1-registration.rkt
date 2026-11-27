#lang racket/base

(require (only-in ffi/vector list->s64vector)
         (only-in racket/list append*)
         (only-in racket/cmdline command-line)
         (only-in racket/match match match-define)
         (only-in racket/promise delay force)
         (only-in torch + narrow randn zeros)
         (only-in torch/foreign/raw/collector wake-after-collection!)
         (only-in torch/foreign/raw/memory account! registering tr-tensor-free/unwrapped)
         (only-in torch/foreign/raw/regions with-uninterruptible)
         (only-in torch/runtime/reaper reap-when-dead! reaped)
         (only-in "harness.rkt" rotate)
         (only-in "flags.rkt" flag-count flag-counts flag-positive require-one-aten-thread!)
         (only-in "m1-clocks.rkt" gc-overlap-ms with-gc-log)
         (only-in "op-cost-measure.rkt" condition condition-layer condition-op sample-condition)
         (only-in "op-cost-ops.rkt"
                  add-scalar/c add/c build-case narrow/c op-case-layer tensors-per-call zeros-on/c))

(provide mix-layer
         mix-layers
         mix-ops
         registration-conditions
         registration-records)

(define mix-layers '(kept raw guardian lifetimes registered accounted facade))

(define mix-ops 4)

(define probe-reaped (reaped 'm1-probe tr-tensor-free/unwrapped))

(define probe-lifetimes (make-weak-hasheq))

(struct probe-lifetime (state charge))

(define kept-batch 16384)

(struct kept-store (handles count) #:mutable)

(struct keeper (batch stores freeing ready pending peak freer))

(define (push! b v)
  (let retry ()
    (define old (unbox b))
    (unless (box-cas! b old (cons v old))
      (retry))))

(define (take-all! b)
  (let retry ()
    (define old (unbox b))
    (if (box-cas! b old '()) old (retry))))

(define (add! b d)
  (let retry ()
    (define old (unbox b))
    (define new (+ old d))
    (if (box-cas! b old new) new (retry))))

(define (raise-to! b v)
  (let retry ()
    (define old (unbox b))
    (unless (or (>= old v) (box-cas! b old v))
      (retry))))

(define (make-keeper #:free [free tr-tensor-free/unwrapped] #:batch [batch kept-batch])
  (define freeing (box '()))
  (define ready (make-semaphore 0))
  (define pending (box 0))
  (define freer
    (delay
      (thread (lambda ()
                (let loop ()
                  (semaphore-wait ready)
                  (for ([item (in-list (reverse (take-all! freeing)))])
                    (match item
                      [(? semaphore? done) (semaphore-post done)]
                      [handles
                       (for-each free handles)
                       (add! pending (- (length handles)))]))
                  (loop)))
              #:pool 'own)))
  (keeper batch (box '()) freeing ready pending (box 0) freer))

(define kept (make-keeper))

(define (post! k item)
  (push! (keeper-freeing k) item)
  (semaphore-post (keeper-ready k)))

(define (hand-off! k store)
  (match-define (kept-store handles count) store)
  (set-kept-store-handles! store '())
  (set-kept-store-count! store 0)
  (raise-to! (keeper-peak k) (add! (keeper-pending k) count))
  (post! k handles))

(define (keeping k)
  (force (keeper-freer k))
  (define store (kept-store '() 0))
  (push! (keeper-stores k) store)
  (lambda (h)
    (set-kept-store-handles! store (cons h (kept-store-handles store)))
    (set-kept-store-count! store (add1 (kept-store-count store)))
    (when (= (kept-store-count store) (keeper-batch k))
      (hand-off! k store))))

(define (free-kept! [k kept])
  (force (keeper-freer k))
  (for ([store (in-list (unbox (keeper-stores k)))])
    (hand-off! k store))
  (define done (make-semaphore 0))
  (post! k done)
  (semaphore-wait done))

(define (guarded h)
  (when h (reap-when-dead! probe-reaped h))
  h)

(define (in-lifetimes h)
  (when h
    (hash-set! probe-lifetimes h (probe-lifetime (box 'live) (box #f)))
    (tr-tensor-free/unwrapped h)))

(define ((waking f) . args)
  (begin0 (apply f args) (wake-after-collection!)))

(define (layered c-call layer keep)
  (match layer
    ['kept
     (lambda args
       (define h (apply c-call args))
       (when h (keep h)))]
    ['raw (lambda args (tr-tensor-free/unwrapped (apply c-call args)))]
    ['guardian (waking (lambda args (with-uninterruptible (guarded (apply c-call args)))))]
    ['lifetimes (lambda args (with-uninterruptible (in-lifetimes (apply c-call args))))]
    ['registered (waking (registering c-call))]
    ['accounted (waking (lambda args
                          (define t (apply (registering c-call) args))
                          (when t (account! t))
                          t))]))

(define (mix-layer layer #:keeper [k kept])
  (define x (randn 3 40 40))
  (define a (randn 8 8))
  (define b (randn 8 8))
  (define one (zeros 1))
  (define dims (list->s64vector '(8 8)))
  (match layer
    ['facade
     (lambda ()
       (narrow x 1 4 32)
       (+ a b)
       (+ one 1.0)
       (zeros 8 8)
       (void))]
    [_
     (define keep (and (eq? layer 'kept) (keeping k)))
     (define narrow* (layered narrow/c layer keep))
     (define add* (layered add/c layer keep))
     (define scalar* (layered add-scalar/c layer keep))
     (define zeros* (layered zeros-on/c layer keep))
     (lambda ()
       (narrow* x 1 4 32)
       (add* a b)
       (scalar* one 1.0)
       (zeros* dims 2 'cpu 0 'float32)
       (void))]))

(define (chain-thunk)
  (op-case-layer (build-case 'cifar-chain) 'facade))

(define (registration-conditions threads #:keeper [k kept])
  (append*
   (for/list ([n (in-list threads)])
     (append (for/list ([l (in-list mix-layers)])
               (condition 'mix l n (lambda () (mix-layer l #:keeper k))))
             (list (condition 'cifar-chain 'facade n chain-thunk))))))

(define (chain-tensors)
  (tensors-per-call (chain-thunk)))

(define (registration-record c seconds rep #:keeper [k kept])
  (set-box! (keeper-peak k) (unbox (keeper-pending k)))
  (define-values (r events) (with-gc-log () (sample-condition c seconds #:rep rep)))
  (define backlog
    (if (eq? (condition-layer c) 'kept)
        (list 'kept_pending_peak (unbox (keeper-peak k)) 'kept_batch (keeper-batch k))
        '()))
  (free-kept! k)
  (apply hash-set* r
         'ops_per_call (if (eq? (condition-op c) 'mix) mix-ops 1)
         'gc_real_ms (gc-overlap-ms events (hash-ref r 'window_start_ms) (hash-ref r 'window_end_ms))
         backlog))

(define (registration-records threads reps seconds #:emit emit)
  (define (emitted r) (emit r) r)
  (cons (emitted (hasheq 'kind "chain-tensors" 'tensors_per_call (exact->inexact (chain-tensors))))
        (for*/list ([rep (in-range reps)] [c (in-list (rotate (registration-conditions threads) rep))])
          (emitted (registration-record c seconds rep)))))

(define (registration-settings argv)
  (define settings (hasheq 'threads '(1 2 4 8) 'reps 3 'seconds 2.0))
  (define (put! key value) (set! settings (hash-set settings key value)))
  (command-line
   #:program "m1-registration"
   #:argv argv
   #:once-each
   [("--threads") s "Comma-separated thread counts (default 1,2,4,8)"
                  (put! 'threads (flag-counts 'm1-registration s exact-positive-integer?))]
   [("--reps") n "Rounds over every condition (default 3)"
               (put! 'reps (flag-count 'm1-registration n))]
   [("--seconds") s "Seconds per condition (default 2)"
                  (put! 'seconds (flag-positive 'm1-registration s))])
  settings)

(module+ main
  (require (only-in json write-json))
  (define settings (registration-settings (current-command-line-arguments)))
  (require-one-aten-thread! 'm1-registration)
  (void (registration-records (hash-ref settings 'threads) (hash-ref settings 'reps)
                              (hash-ref settings 'seconds)
                              #:emit (lambda (r) (write-json (hash-set r 'racket (version))) (newline)))))

(module+ test
  (require rackunit
           (only-in "op-cost-ops.rkt" crossings-per-call)
           (only-in "summary-m1.rkt" m1-markdown registration-rows))

  (test-case "every mix layer crosses once per op, and only the registering layers leave tensors"
    (for ([l (in-list mix-layers)] #:unless (eq? l 'facade))
      (check-true (>= (crossings-per-call (mix-layer l)) mix-ops) (symbol->string l)))
    (free-kept!)
    (check-equal? (for/list ([l (in-list mix-layers)]) (cons l (tensors-per-call (mix-layer l))))
                  `((kept . 0) (raw . 0) (guardian . 0) (lifetimes . 0) (registered . ,mix-ops)
                    (accounted . ,mix-ops) (facade . ,mix-ops)))
    (free-kept!))

  (test-case "the kept layer hands its handles off the calling thread in batches, the rest at the end"
    (define k (make-keeper #:batch (* 2 mix-ops)))
    (define call (mix-layer 'kept #:keeper k))
    (define (held) (for/sum ([s (in-list (unbox (keeper-stores k)))]) (kept-store-count s)))
    (call)
    (check-equal? (held) mix-ops)
    (call)
    (check-equal? (held) 0 "a full batch went to the freeing thread")
    (call)
    (free-kept! k)
    (check-equal? (held) 0)
    (check-equal? (unbox (keeper-pending k)) 0)
    (check-equal? (unbox (keeper-freeing k)) '()))

  (test-case "a store free-kept! emptied hands each handle to the freer once, the next batch too"
    (define seen (make-hasheq))
    (define (free h)
      (hash-update! seen h add1 0)
      (when (= (hash-ref seen h) 1)
        (tr-tensor-free/unwrapped h)))
    (define k (make-keeper #:batch (* 2 mix-ops) #:free free))
    (define call (mix-layer 'kept #:keeper k))
    (call)
    (free-kept! k)
    (for ([_ (in-range 3)]) (call))
    (free-kept! k)
    (check-equal? (hash-count seen) (* 4 mix-ops) "every kept handle reached the freer")
    (check-equal? (filter (lambda (n) (> n 1)) (hash-values seen)) '() "and none twice"))

  (test-case "a freer that falls behind records its backlog, and the summary flags the free piece"
    (define batch (* 2 mix-ops))
    (define (stalled-free h)
      (let stall ()
        (when (<= (unbox (keeper-peak k)) (* 3 batch))
          (sleep 0.001)
          (stall)))
      (tr-tensor-free/unwrapped h))
    (define k (make-keeper #:batch batch #:free stalled-free))
    (define records
      (for/list ([c (in-list (registration-conditions '(1) #:keeper k))]
                 #:when (memq (condition-layer c) '(raw kept)))
        (registration-record c 0.1 0 #:keeper k)))
    (match-define (list kept-record) (filter (lambda (r) (equal? (hash-ref r 'layer) "kept")) records))
    (define peak (hash-ref kept-record 'kept_pending_peak))
    (check-true (> peak (* 3 batch)) (format "peak ~a" peak))
    (check-equal? (hash-ref kept-record 'kept_batch) batch)
    (define-values (_threads _layers _pieces _chain backlog) (registration-rows records))
    (check-equal? backlog (list (cons peak #t)))
    (check-regexp-match #rx"\\| free \\| [^|]* \\(backlog\\) \\|" (m1-markdown records)))

  (test-case "settings read counts and seconds, and refuse what is not one"
    (check-equal? (registration-settings (vector "--threads" "1,4" "--seconds" "0.5"))
                  #hasheq((threads . (1 4)) (reps . 3) (seconds . 0.5)))
    (check-exn #rx"expected a count" (lambda () (registration-settings (vector "--threads" "0"))))
    (check-exn #rx"expected a positive" (lambda () (registration-settings (vector "--seconds" "x")))))

  (test-case "a condition per layer and thread count, and the chain's, at tiny sizes"
    (check-equal? (length (registration-conditions '(1 2))) (* 2 (add1 (length mix-layers))))
    (match-define (cons chain ops) (registration-records '(1) 2 0.1 #:emit void))
    (check-true (positive? (hash-ref chain 'tensors_per_call)))
    (for ([r (in-list ops)])
      (check-true (positive? (hash-ref r 'ops)) (hash-ref r 'layer))
      (check-true (<= 0 (hash-ref r 'gc_real_ms) (hash-ref r 'elapsed_ms))
                  "only collections inside the timed window count"))
    (define-values (_threads _layers pieces _chain _backlog) (registration-rows ops))
    (for ([p (in-list pieces)])
      (match-define (list name (list us free kb)) p)
      (check-true (andmap real? (list us free kb)) (format "~a has its two layers" name)))))
