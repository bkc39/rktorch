#lang racket/base

;; I1's cases stay in the order of the twin, torch/tests/python/trunc_normal.py.

(module+ test
  (require rackunit
           (only-in racket/contract exn:fail:contract:blame?)
           (only-in racket/match match-define)
           (only-in "../main.rkt" in-flattened-tensor manual-seed! randn tensor->list)
           (only-in "../nn.rkt" trunc-normal-init uniform-init)
           (only-in "private/python-env.rkt" python-check python-torch-available?)
           (only-in "private/within.rkt" bounded-test))

  (define (flat t)
    (for/list ([v (in-flattened-tensor t)]) v))

  (define (seeded thunk)
    (manual-seed! 3)
    (thunk))

  (define cases
    (list (cons 'std (lambda () (trunc-normal-init '(64 32) #:std 0.02)))
          (cons 'bounds
                (lambda ()
                  (trunc-normal-init '(64 32) #:mean 0.01 #:std 0.02
                                     #:a -0.01 #:b 0.03)))))

  (cond
    [(python-torch-available?)
     (bounded-test "I1: trunc-normal-init draws the inverse-CDF truncated normal's values"
       (define j (python-check "trunc_normal.py"))
       (for ([named (in-list cases)])
         (match-define (cons key run) named)
         (define got (flat (seeded run)))
         (define expected (hash-ref j key))
         (check-equal? (length got) (length expected))
         (for ([a (in-list got)] [b (in-list expected)] [i (in-naturals)])
           (check-= a b 1e-6 (format "~a ~a" key i)))))]
    [else
     (printf "[trunc-normal-test] skipped I1: python3 `torch` ~a\n"
             "not available (run inside `nix develop`)")])

  (bounded-test "I2: every value lies in [a, b]"
    (for ([bounds (in-list '((-0.01 0.03) (-2.0 2.0) (0 1/2)))])
      (match-define (list a b) bounds)
      (define t (seeded (lambda ()
                          (trunc-normal-init '(16 8) #:mean 0.01 #:std 0.5
                                             #:a a #:b b))))
      (for ([v (in-list (flat t))])
        (check-true (<= a v b) (format "~a in ~a" v bounds)))))

  (bounded-test "I2: the draw advances the stream as uniform-init does"
    (define after-trunc
      (seeded (lambda ()
                (trunc-normal-init '(7 5) #:std 0.3)
                (tensor->list (randn 4)))))
    (define after-uniform
      (seeded (lambda ()
                (uniform-init '(7 5) -0.25 0.75)
                (tensor->list (randn 4)))))
    (check-equal? after-trunc after-uniform))

  (bounded-test "trunc-normal-init blames a caller whose #:a is not below #:b"
    (check-exn (lambda (e)
                 (and (exn:fail:contract:blame? e)
                      (regexp-match? #rx"blaming: [^\n]*trunc-normal-test"
                                     (exn-message e))))
               (lambda () (trunc-normal-init '(2 2) #:a 1.0 #:b 1.0)))))
