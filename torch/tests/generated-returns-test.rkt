#lang racket/base

(module+ test
  (require rackunit
           (only-in syntax/macro-testing convert-syntax-error)
           (only-in "../generated.rkt" sort-tensor topk)
           "../main.rkt"
           (only-in "../foreign/define-generated.rkt" define-generated-op)
           (only-in "../foreign/raw/pressure.rkt" reset-pressure-state!)
           (submod "../foreign.rkt" unsafe))

  (define (cpu-bytes)
    (for/sum ([entry (in-list (native-memory-use))]
              #:when (eq? (device-type (car entry)) 'cpu))
      (cdr entry)))

  (test-case "topk answers values and indices as two values"
    (define-values (values-t indices-t)
      (topk (tensor '((1.0 5.0 3.0) (4.0 2.0 6.0))) 2 -1 #t #t))
    (check-equal? (tensor-shape values-t) '(2 2))
    (check-equal? (tensor->list values-t) '(5.0 3.0 6.0 4.0))
    (check-equal? (tensor-dtype indices-t) 'int64)
    (check-equal? (tensor->list indices-t) '(1 2 2 0)))

  (test-case "sort-tensor answers the sorted values and their source indices"
    (define-values (values-t indices-t)
      (sort-tensor (tensor '(3.0 1.0 2.0)) 0 #t))
    (check-equal? (tensor->list values-t) '(3.0 2.0 1.0))
    (check-equal? (tensor->list indices-t) '(0 2 1)))

  (test-case "gradients flow through the values output"
    (define x (requires-grad! (tensor '(1.0 5.0 3.0))))
    (define-values (top _indices) (topk x 2 0 #t #t))
    (backward! (sum top))
    (check-equal? (tensor->list (grad x)) '(0.0 1.0 1.0)))

  (test-case "the ledger accounts every output and releases each on free"
    (define input (randn 64))
    (define before (cpu-bytes))
    (define-values (values-t indices-t) (topk input 16 0 #t #t))
    (check-equal? (- (cpu-bytes) before) (+ (* 16 4) (* 16 8)))
    (tensor-free! values-t)
    (check-equal? (- (cpu-bytes) before) (* 16 8))
    (tensor-free! indices-t)
    (check-equal? (cpu-bytes) before))

  ;; Only the two outputs are allocated inside the loop, so the trigger can
  ;; only come from the multi-output wrap.
  (test-case "a multi-output call presses on the collector like any other"
    (define mib (* 1024 1024))
    (define (collections)
      (cdr (assq 'pressure-collections (finalizer-diagnostics))))
    (define source (randn 1048576))
    (for ([_ (in-range 3)]) (reclaim-native-memory!))
    (reset-pressure-state!)
    (define base (cpu-bytes))
    (define before (collections))
    (parameterize ([native-memory-limit (* 32 mib)])
      (define high-water
        (for/fold ([hw 0]) ([_ (in-range 40)])
          (define-values (top indices) (topk source 262144 0 #t #t))
          (void top indices)
          (max hw (- (cpu-bytes) base))))
      (check-true (> (collections) before)
                  "accounting several outputs never reached the trigger")
      (check-true (< high-water (* 64 mib))
                  (format "ledger reached ~a MiB under a 32 MiB limit"
                          (quotient high-water mib))))
    (check-equal? (tensor-shape source) '(1048576)))

  (test-case "a failing call raises with the op and the ATen message"
    (check-exn #rx"topk.*tr_gen_topk"
               (lambda () (topk (tensor '(1.0 2.0 3.0)) 4 0 #t #t))))

  (test-case "#:returns below 2 is a syntax error"
    (check-exn
     #rx"a single return takes no #:returns clause"
     (lambda ()
       (convert-syntax-error
        (let ()
          (define-generated-op one tr_gen_topk #:returns 1 ([self tensor]))
          (void)))))))
