#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in ffi/unsafe
                  _double _enum _fun _int _int64 _ptr _void
                  register-finalizer)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic call-as-atomic in-atomic-mode?)
         (only-in "../device-type.rkt" device device-index device-type)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-tensor-free/finalizer
         tr-tensor-free/checked
         collect-and-drain!
         collect-at-trough!
         swallow-and-count-failure
         finalizer-failures
         finalizer-diagnostics
         tensor-allocator
         tensor-allocator/rng
         oom-retry
         oom-retry/status
         reaccount!
         tr-cuda-empty-cache/raw
         tr-cuda-mem-get-info/raw
         tr-cuda-memory-stats/raw
         tr-mps-empty-cache/raw
         tr-last-error-kind/raw
         allocator-reading
         native-memory-limit
         native-memory-use
         native-memory-use/fold
         reset-pressure-state!
         trough-budget
         trough-margin
         _tr-device-type ;; noqa
         tr-tensor-device/raw
         define-unary/raw
         define-binary/raw
         define-scalar/raw)

(define-torch tr-tensor-free/unwrapped
  (_fun _Tensor -> _void)
  #:c-id tr_tensor_free)

;; The (deallocator) wrap cancels the pending GC finalizer.
(define tr-tensor-free/checked
  (let ([release ((deallocator) tr-tensor-free/unwrapped)])
    (lambda (t)
      (unaccount! t)
      (release t))))

(define finalizer-failure-count (box 0))
(define finalizer-run-count (box 0))
(define captured-failures (box '()))
(define capture-limit 8)
(define pressure-collection-count (box 0))
(define pressure-reclaimed-bytes (box 0))
(define trough-collection-count (box 0))
(define next-trough-ms (box 0.0))

(define (finalizer-failures)
  (unbox finalizer-failure-count))

(define (finalizer-diagnostics)
  (call-with-ledger
   (lambda ()
     (list (cons 'runs (unbox finalizer-run-count))
           (cons 'failures (unbox finalizer-failure-count))
           (cons 'messages (reverse (unbox captured-failures)))
           (cons 'ledger-entries (hash-count allocations))
           (cons 'pressure-collections (unbox pressure-collection-count))
           (cons 'pressure-reclaimed (unbox pressure-reclaimed-bytes))
           (cons 'trough-collections (unbox trough-collection-count))))))

;; No printer: a prop:custom-write that raises or blocks would be fatal here.
(define (describe-raised e)
  (cond
    [(exn? e) (exn-message e)]
    [(symbol? e) (symbol->string e)]
    [(string? e) e]
    [else "non-exn value raised from a native release"]))

(define (take-at-most n xs)
  (cond
    [(or (zero? n) (null? xs)) '()]
    [else (cons (car xs) (take-at-most (sub1 n) (cdr xs)))]))

;; Guarded: this is the handler below, so nothing else protects it.
(define (record-failure! e)
  (with-handlers ([(lambda (_) #t) void])
    (record-failure!/unguarded e)))

(define (record-failure!/unguarded e)
  (call-with-ledger
   (lambda ()
     (set-box! finalizer-failure-count (add1 (unbox finalizer-failure-count)))
     (set-box! captured-failures
               (take-at-most capture-limit
                             (cons (describe-raised e)
                                   (unbox captured-failures)))))))

;; Total catch on purpose, not exn:fail?: any value escaping GC
;; finalization re-enters the error machinery and cascades (#38).
(define ((swallow-and-count-failure release) t)
  ;; Nothing outside the guard: alloc.rkt's finalizer holds raw atomic mode
  ;; with no dynamic-wind, so an escape kills the process rather than raising.
  (with-handlers ([(lambda (_) #t) record-failure!])
    (call-with-ledger
     (lambda () (set-box! finalizer-run-count (add1 (unbox finalizer-run-count)))))
    (release t)))

(struct allocation (phantom nbytes device))

(define allocations (make-weak-hasheq))
(define live-bytes (make-hash))
(define accounted-since (make-hash))
(define sampled-since (make-hash))
(define allocator-sample (make-hash))
(define collect-interval (make-hash))
(define high-water (make-hash))
(define trough-floor (make-hash))

(define native-memory-limit (make-parameter #f))
(define high-water-fraction 4/5)
(define interval-divisor 8)
(define sample-divisor 32)
(define reclaim-fraction 1/20)

;; #f: the floor itself, kept between the two bounds below
(define trough-margin (make-parameter #f))
(define trough-margin-min (* 256 1024 1024))
(define trough-margin-max (* 1024 1024 1024))

;; the share of wall-clock time trough collections may take
(define trough-budget (make-parameter 1/20))

;; Atomic mode, not a semaphore: finalizers run in atomic mode, where
;; blocking is an internal error.
(define (call-with-ledger thunk)
  (call-as-atomic thunk))

(define-torch tr-tensor-nbytes/raw
  (_fun _Tensor (out : (_ptr o _int64)) -> (rc : _int) -> (values rc out))
  #:c-id tr_tensor_nbytes)

(define _tr-device-type
  (_enum '(cpu = 0 cuda = 1 mps = 2 keep = -1) _int))

(define-torch tr-tensor-device/raw
  (_fun _Tensor
        (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_tensor_device)

(define-torch tr-cuda-mem-get-info/raw
  (_fun (index : _int64)
        (free : (_ptr o _int64))
        (total : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc free total))
  #:c-id tr_cuda_mem_get_info)

(define-torch tr-cuda-memory-stats/raw
  (_fun (index : _int64)
        (allocated : (_ptr o _int64))
        (reserved : (_ptr o _int64))
        (peak : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc allocated reserved peak))
  #:c-id tr_cuda_memory_stats)

(define (account! t)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (define-values (nb-rc nbytes) (tr-tensor-nbytes/raw t))
    (define-values (dev-rc type index) (tr-tensor-device/raw t))
    (and (zero? nb-rc)
         (zero? dev-rc)
         (let* ([dev (device type (if (eq? type 'cpu) 0 index))]
                [entry (allocation (make-phantom-bytes nbytes) nbytes dev)])
           (call-with-ledger
            (lambda ()
              (hash-set! allocations t entry)
              (hash-update! live-bytes dev (lambda (n) (+ n nbytes)) 0)
              (hash-update! accounted-since dev (lambda (n) (+ n nbytes)) 0)
              (hash-update! sampled-since dev (lambda (n) (+ n nbytes)) 0)))
           dev))))

(define (unaccount! t)
  (with-handlers ([exn:fail? void])
    (call-with-ledger
     (lambda ()
       (define a (hash-ref allocations t #f))
       (when a
         (set-phantom-bytes! (allocation-phantom a) 0)
         (hash-remove! allocations t)
         (hash-update! live-bytes
                       (allocation-device a)
                       (lambda (n) (max 0 (- n (allocation-nbytes a))))
                       0))))))

;; An in-place move (tr_tensor_to_) changes the device and byte count under
;; the same handle, so its ledger entry is replaced rather than added to.
(define (reaccount! t)
  (unaccount! t)
  (define dev (account! t))
  (when dev
    (collect-under-pressure! dev)))

(define (sort-by-device totals)
  (sort totals
        (lambda (x y)
          (define dx (car x))
          (define dy (car y))
          (cond
            [(eq? (device-type dx) (device-type dy))
             (< (device-index dx) (device-index dy))]
            [else (eq? (device-type dx) 'cpu)]))))

(define (native-memory-use)
  (define totals (call-with-ledger (lambda () (hash->list live-bytes))))
  (sort-by-device (filter (lambda (entry) (positive? (cdr entry))) totals)))

;; The entry-by-entry fold; the counters above must agree with it.
(define (native-memory-use/fold)
  (define entries (call-with-ledger (lambda () (hash-values allocations))))
  (define totals (make-hash))
  (for ([a (in-list entries)])
    (hash-update! totals (allocation-device a)
                  (lambda (n) (+ n (allocation-nbytes a)))
                  0))
  (sort-by-device (hash->list totals)))

(define (capacity-mark dev)
  (cond
    [(eq? (device-type dev) 'cuda)
     (define-values (rc _free total)
       (tr-cuda-mem-get-info/raw (device-index dev)))
     (and (zero? rc) (positive? total) (floor (* high-water-fraction total)))]
    [else #f]))

;; Queried once per device, outside atomic mode; #f when unknown.
(define (device-high-water dev)
  (or (native-memory-limit)
      (let ([cached (call-with-ledger
                     (lambda () (hash-ref high-water dev 'unknown)))])
        (cond
          [(eq? cached 'unknown)
           (define mark (capacity-mark dev))
           (call-with-ledger (lambda () (hash-set! high-water dev mark)))
           mark]
          [else cached]))))

;; The allocator's own allocated bytes: the ledger double-counts views and
;; cannot see storage only the autograd graph holds. #f when unknown.
(define (cuda-allocated dev)
  (cond
    [(eq? (device-type dev) 'cuda)
     (define-values (rc allocated _reserved _peak)
       (tr-cuda-memory-stats/raw (device-index dev)))
     (and (zero? rc) allocated)]
    [else #f]))

(define allocator-reading (make-parameter cuda-allocated))

(define (sample-allocator! dev)
  (define reading (or ((allocator-reading) dev) 0))
  (call-with-ledger
   (lambda ()
     (hash-set! allocator-sample dev reading)
     (hash-set! sampled-since dev 0)))
  reading)

(define (collect-under-pressure! dev)
  (unless (in-atomic-mode?)
    (define mark (device-high-water dev))
    (when mark
      (define base (quotient mark interval-divisor))
      (define-values (gate-open? sample-due?)
        (call-with-ledger
         (lambda ()
           (values (>= (hash-ref accounted-since dev 0)
                       (hash-ref collect-interval dev base))
                   (>= (hash-ref sampled-since dev 0)
                       (quotient mark sample-divisor))))))
      (when gate-open?
        (when sample-due?
          (sample-allocator! dev))
        (when (> (pressure-reading dev) mark)
          (pressure-collect! dev mark base))))))

(define (pressure-reading dev)
  (call-with-ledger
   (lambda ()
     (max (hash-ref live-bytes dev 0)
          (hash-ref allocator-sample dev 0)))))

(define (reset-pressure-state!)
  (call-with-ledger
   (lambda ()
     (for ([table (in-list (list accounted-since sampled-since allocator-sample
                                 collect-interval high-water trough-floor))])
       (hash-clear! table))
     (set-box! next-trough-ms 0.0))))

;; Reclaiming little means the working set itself sits above the mark, so
;; the interval to the next collection doubles instead of thrashing.
(define (pressure-collect! dev mark base)
  (define before (pressure-reading dev))
  (collect-and-wait!)
  (sample-allocator! dev)
  (define after (pressure-reading dev))
  (call-with-ledger
   (lambda ()
     (define reclaimed (max 0 (- before after)))
     (define interval (hash-ref collect-interval dev base))
     (hash-set! collect-interval dev
                (if (< reclaimed (* reclaim-fraction mark))
                    (min (* 2 interval) (* 2 mark))
                    base))
     (hash-set! accounted-since dev 0)
     (set-box! pressure-collection-count
               (add1 (unbox pressure-collection-count)))
     (set-box! pressure-reclaimed-bytes
               (+ reclaimed (unbox pressure-reclaimed-bytes))))))

;; The end of backward! is a step's trough: the graph is released, the
;; forward's intermediates are dead, and little is live, so a collection
;; here reclaims the most, promotes the least, and leaves Racket's own
;; schedule a low baseline. The floor is the ledger's size after the last
;; collection at a trough, and residue past the margin above it is due one.
;; A step's dead intermediates have aged past the nursery by then, so only
;; a full collection reaches them, and the budget spaces those out: after
;; one that took t, the next waits t over the budget.
(define (margin-over floor)
  (or (trough-margin)
      (max trough-margin-min (min floor trough-margin-max))))

(define (devices-over-floor)
  (call-with-ledger
   (lambda ()
     (for/list ([(dev live) (in-hash live-bytes)]
                #:when (let ([floor (hash-ref trough-floor dev #f)])
                         (and floor (> (- live floor) (margin-over floor)))))
       dev))))

(define (set-trough-floors!)
  (call-with-ledger
   (lambda ()
     (for ([(dev live) (in-hash live-bytes)])
       (hash-set! trough-floor dev live)))))

(define (lower-trough-floors!)
  (call-with-ledger
   (lambda ()
     (for ([(dev live) (in-hash live-bytes)])
       (hash-update! trough-floor dev (lambda (floor) (min floor live)) live)))))

(define (ledger-total)
  (call-with-ledger
   (lambda ()
     (for/sum ([live (in-hash-values live-bytes)]) live))))

(define (collect-at-trough!)
  (unless (in-atomic-mode?)
    (lower-trough-floors!)
    (define started (current-inexact-milliseconds))
    (when (and (>= started (unbox next-trough-ms))
               (pair? (devices-over-floor)))
      (define before (ledger-total))
      (collect-and-wait!)
      (set-trough-floors!)
      (call-with-ledger
       (lambda ()
         (for ([dev (in-list (hash-keys accounted-since))])
           (hash-set! accounted-since dev 0))))
      (define finished (current-inexact-milliseconds))
      (set-box! next-trough-ms
                (+ finished (/ (- finished started) (trough-budget))))
      (set-box! trough-collection-count
                (add1 (unbox trough-collection-count)))
      (set-box! pressure-reclaimed-bytes
                (+ (max 0 (- before (ledger-total)))
                   (unbox pressure-reclaimed-bytes))))))

;; Unwrapped release on purpose: the (deallocator) wrap would cancel the
;; very registration this finalizer runs from.
(define tr-tensor-free/finalizer
  (swallow-and-count-failure
   (lambda (t)
     (unaccount! t)
     (tr-tensor-free/unwrapped t))))

(define-torch tr-last-error-kind/raw
  (_fun -> _int)
  #:c-id tr_last_error_kind)

(define (last-error-oom?)
  (= 1 (tr-last-error-kind/raw)))

(define-torch tr-cuda-empty-cache/raw
  (_fun -> _int)
  #:c-id tr_cuda_empty_cache)

(define-torch tr-mps-empty-cache/raw
  (_fun -> _int)
  #:c-id tr_mps_empty_cache)

;; The canary shows the finalizer thread has started on this collection's
;; batch, not that it has finished: finalization order is unspecified and
;; the thread runs only while this one yields. So yield until the run count
;; has stood still for a few turns, within a deadline.
(define drain-deadline-ms 2000)
(define quiet-turns 3)

(define (collect-and-wait!)
  (define canary-finalized (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post canary-finalized)))
  (collect-garbage)
  (define observed (sync/timeout 0.5 canary-finalized))
  (drain-finalizers!)
  (and observed #t))

(define (drain-finalizers!)
  (define deadline (+ (current-inexact-milliseconds) drain-deadline-ms))
  (let loop ([runs (unbox finalizer-run-count)] [quiet 0])
    (sleep 0)
    (define now (unbox finalizer-run-count))
    (cond
      [(> (current-inexact-milliseconds) deadline) (void)]
      [(not (= now runs)) (loop now 0)]
      [(< (add1 quiet) quiet-turns) (loop now (add1 quiet))]
      [else (void)])))

(define (collect-and-drain!)
  (define observed (collect-and-wait!))
  (void (tr-cuda-empty-cache/raw))
  (void (tr-mps-empty-cache/raw))
  observed)

;; one retry after a collect when a failed call was an OOM; the two
;; wrappers below differ only in how a raw result reports failure
(define (((retry-on-oom failed? oom? collect!) raw-fn) . args)
  (define result (apply raw-fn args))
  (cond
    [(and (failed? result) (oom?))
     (collect!)
     (apply raw-fn args)]
    [else result]))

;; handle-returning raw calls: #f is the failure
(define (oom-retry #:oom? [oom? last-error-oom?]
                   #:collect! [collect! collect-and-drain!])
  (retry-on-oom not oom? collect!))

;; status-returning raw calls: 1 is the failure
(define (oom-retry/status #:oom? [oom? last-error-oom?]
                          #:collect! [collect! collect-and-drain!])
  (retry-on-oom (lambda (rc) (= rc 1)) oom? collect!))

(define ((accounted wrapped) . args)
  (define t (apply wrapped args))
  (when t
    (define dev (account! t))
    (when dev
      (collect-under-pressure! dev)))
  t)

;; The retry composes OUTSIDE the allocator wrap: ffi/unsafe/alloc runs
;; the wrapped call in atomic mode, where the drain's blocking wait is an
;; internal error.
(define (tensor-allocator raw-fn)
  (accounted ((oom-retry) ((allocator tr-tensor-free/finalizer) raw-fn))))

;; No retry: these bindings consume the global RNG stream, and a blind
;; retry would draw twice and break seeded parity.
(define (tensor-allocator/rng raw-fn)
  (accounted ((allocator tr-tensor-free/finalizer) raw-fn)))

(define-syntax (define-unary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (t : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

(define-syntax (define-binary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

(define-syntax (define-scalar/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _double) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

;; No I/O from inside a finalizer -- that runs in atomic mode, where writing to
;; a port can block and trip an internal error.  Accumulate, dump here.
(define mem-trace-handle
  (and (getenv "RKTORCH_MEM_TRACE")
       (plumber-add-flush!
        (current-plumber)
        (lambda (_h)
          (with-handlers ([(lambda (_) #t) void])
            (eprintf "[rktorch mem] ~s\n" (finalizer-diagnostics)))))))
