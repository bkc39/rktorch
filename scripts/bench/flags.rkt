#lang racket/base

(require (only-in racket/match match)
         (only-in racket/string string-split)
         (only-in "op-cost.rkt" inside-repo?)
         (only-in "shim.rkt" aten-threads))

(provide flag-count
         flag-counts
         flag-positive
         flag-trips
         refuse-repo-out-dir!
         require-one-aten-thread!)

(define (flag-count who text [ok? exact-positive-integer?])
  (define n (string->number text))
  (unless (ok? n)
    (raise-user-error who "expected a count, got ~s" text))
  n)

(define (flag-counts who text [ok? exact-nonnegative-integer?])
  (map (lambda (t) (flag-count who t ok?)) (string-split text ",")))

(define (flag-positive who text)
  (define x (string->number text))
  (unless (and (real? x) (< 0 x +inf.0))
    (raise-user-error who "expected a positive number, got ~s" text))
  x)

(define (flag-trips who text)
  (for/list ([t (in-list (string-split text ","))])
    (match t
      ["-" #f]
      [_ (flag-positive who t)])))

(define (refuse-repo-out-dir! who dir root)
  (when (inside-repo? dir root)
    (raise-user-error who "~a is inside the repository; bench results are never committed" dir)))

(define (require-one-aten-thread! who)
  (define aten (aten-threads))
  (unless (eqv? aten 1)
    (raise-user-error who "needs one ATen intra-op thread, got ~a: set OMP_NUM_THREADS=1" aten))
  aten)

(module+ test
  (require rackunit)

  (test-case "counts, positives and trips read their text or name it"
    (check-equal? (flag-counts 'probe "0,7") '(0 7))
    (check-equal? (flag-trips 'probe "-,32,0.5") '(#f 32 0.5))
    (check-exn #rx"probe: expected a count, got \"0\"" (lambda () (flag-count 'probe "0")))
    (check-exn #rx"expected a positive number, got \"-1\"" (lambda () (flag-trips 'probe "-1")))
    (check-exn #rx"expected a positive number" (lambda () (flag-positive 'probe "+inf.0"))))

  (test-case "a run needs one ATen intra-op thread, and names the setting otherwise"
    (if (eqv? (aten-threads) 1)
        (check-equal? (require-one-aten-thread! 'probe) 1)
        (check-exn #rx"set OMP_NUM_THREADS=1" (lambda () (require-one-aten-thread! 'probe))))))
