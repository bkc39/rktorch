#lang racket/base

(require (only-in ffi/vector list->s64vector)
         (only-in racket/list append*)
         (only-in racket/match match)
         (only-in torch + narrow randn zeros)
         (only-in torch/foreign/raw/collector wake-after-collection!)
         (only-in torch/foreign/raw/memory account! registering tr-tensor-free/unwrapped)
         (only-in torch/foreign/raw/regions with-uninterruptible)
         (only-in torch/runtime/reaper reap-when-dead! reaped)
         (only-in "harness.rkt" rotate)
         (only-in "m1-clocks.rkt" call-with-gc-log gc-event-end gc-event-start)
         (only-in "op-cost-measure.rkt" condition condition-op sample-condition)
         (only-in "op-cost-ops.rkt"
                  add-scalar/c add/c build-case narrow/c op-case-layer tensors-per-call zeros-on/c))

(provide mix-layer
         mix-layers
         mix-ops
         registration-conditions)

(define mix-layers '(raw guardian lifetimes registered accounted facade))

(define mix-ops 4)

(define probe-reaped (reaped 'm1-probe tr-tensor-free/unwrapped))

(define probe-lifetimes (make-weak-hasheq))

(struct probe-lifetime (state charge))

(define (guarded h)
  (when h (reap-when-dead! probe-reaped h))
  h)

(define (in-lifetimes h)
  (when h
    (hash-set! probe-lifetimes h (probe-lifetime (box 'live) (box #f)))
    (tr-tensor-free/unwrapped h)))

(define ((waking f) . args)
  (begin0 (apply f args) (wake-after-collection!)))

(define (layered c-call layer)
  (match layer
    ['raw (lambda args (tr-tensor-free/unwrapped (apply c-call args)))]
    ['guardian (waking (lambda args (with-uninterruptible (guarded (apply c-call args)))))]
    ['lifetimes (lambda args (with-uninterruptible (in-lifetimes (apply c-call args))))]
    ['registered (waking (registering c-call))]
    ['accounted (waking (lambda args
                          (define t (apply (registering c-call) args))
                          (when t (account! t))
                          t))]))

(define (mix-layer layer)
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
     (define narrow* (layered narrow/c layer))
     (define add* (layered add/c layer))
     (define scalar* (layered add-scalar/c layer))
     (define zeros* (layered zeros-on/c layer))
     (lambda ()
       (narrow* x 1 4 32)
       (add* a b)
       (scalar* one 1.0)
       (zeros* dims 2 'cpu 0 'float32)
       (void))]))

(define (chain-thunk)
  (op-case-layer (build-case 'cifar-chain) 'facade))

(define (registration-conditions threads)
  (append*
   (for/list ([n (in-list threads)])
     (append (for/list ([l (in-list mix-layers)])
               (condition 'mix l n (lambda () (mix-layer l))))
             (list (condition 'cifar-chain 'facade n chain-thunk))))))

(define (chain-tensors)
  (tensors-per-call (chain-thunk)))

(define (registration-records threads reps seconds #:emit emit)
  (emit (hasheq 'kind "chain-tensors" 'tensors_per_call (exact->inexact (chain-tensors))))
  (define conds (registration-conditions threads))
  (for* ([rep (in-range reps)] [c (in-list (rotate conds rep))])
    (define-values (r events) (call-with-gc-log (lambda () (sample-condition c seconds #:rep rep))))
    (emit (hash-set* r
                     'ops_per_call (if (eq? (condition-op c) 'mix) mix-ops 1)
                     'gc_real_ms (for/sum ([e (in-list events)])
                                   (- (gc-event-end e) (gc-event-start e)))))))

(module+ main
  (require (only-in json write-json)
           (only-in racket/cmdline command-line)
           (only-in racket/string string-split))
  (define threads '(1 2 4 8))
  (define reps 3)
  (define seconds 2.0)
  (command-line
   #:once-each
   [("--threads") s "Comma-separated thread counts (default 1,2,4,8)"
                  (set! threads (map string->number (string-split s ",")))]
   [("--reps") n "Rounds over every condition (default 3)" (set! reps (string->number n))]
   [("--seconds") s "Seconds per condition (default 2)" (set! seconds (string->number s))])
  (registration-records threads reps seconds
                        #:emit (lambda (r) (write-json (hash-set r 'racket (version))) (newline))))

(module+ test
  (require rackunit)

  (test-case "every mix layer runs its four ops, and only the registering layers leave tensors"
    (for ([l (in-list mix-layers)])
      (check-equal? ((mix-layer l)) (void) (symbol->string l)))
    (check-equal? (tensors-per-call (mix-layer 'raw)) 0)
    (check-equal? (tensors-per-call (mix-layer 'guardian)) 0)
    (check-equal? (tensors-per-call (mix-layer 'registered)) mix-ops)
    (check-equal? (tensors-per-call (mix-layer 'accounted)) mix-ops))

  (test-case "a condition per layer and thread count, and the chain's, at tiny sizes"
    (define conds (registration-conditions '(1 2)))
    (check-equal? (length conds) (* 2 (add1 (length mix-layers))))
    (define rs '())
    (registration-records '(1) 1 0.05 #:emit (lambda (r) (set! rs (cons r rs))))
    (check-true (positive? (hash-ref (car (reverse rs)) 'tensors_per_call)))
    (for ([r (in-list (cdr (reverse rs)))])
      (check-true (positive? (hash-ref r 'ops)) (hash-ref r 'layer)))))
