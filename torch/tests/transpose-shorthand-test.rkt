#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt")

  (test-case "T reverses matrix and higher-rank axes"
    (define matrix (tensor '((1 2 3) (4 5 6))))
    (check-equal? (shape (T matrix)) '(3 2))
    (check-equal? (tensor->list (T matrix)) '(1 4 2 5 3 6))
    (check-equal? (dtype (T matrix)) 'int64)
    (check-equal? (tensor->list (~> matrix T)) '(1 4 2 5 3 6))
    (define cube (reshape (arange 12) 2 2 3))
    (check-equal? (shape (T cube)) '(3 2 2))
    (check-equal? (tensor->list (T cube))
                  '(0.0 6.0 3.0 9.0 1.0 7.0 4.0 10.0 2.0 8.0 5.0 11.0))
    (check-equal? (shape (T (zeros 2 3 4 5))) '(5 4 3 2)))

  (test-case "T supports scalar, vector, and empty tensors"
    (for ([x (in-list (list (tensor 7) (tensor '(1 2 3)) (zeros 2 0 3)))])
      (check-equal? (shape (T x)) (reverse (shape x)))
      (check-equal? (tensor->list (T (T x))) (tensor->list x))))

  (test-case "T shares storage and preserves autograd"
    (define source (tensor '((1.0 2.0 3.0) (4.0 5.0 6.0))))
    (zero! (T source))
    (check-equal? (tensor->list source) '(0.0 0.0 0.0 0.0 0.0 0.0))
    (define x (tensor '((1.0 2.0 3.0) (4.0 5.0 6.0)) #:requires-grad? #t))
    (backward! (sum (* (T x) (tensor '((1.0 4.0) (2.0 5.0) (3.0 6.0))))))
    (check-equal? (tensor->list (grad x)) '(1.0 2.0 3.0 4.0 5.0 6.0)))

  (test-case "T rejects non-tensors at the contract boundary"
    (check-exn exn:fail:contract? (lambda () (T 3)))))
