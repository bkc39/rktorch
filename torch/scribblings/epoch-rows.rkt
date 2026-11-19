#lang racket/base

(require racket/contract/base
         (only-in racket/format ~a)
         (only-in racket/match match-define)
         (only-in "../private/contract.rkt" define/contract-out))

(define epoch-row/c
  (list/c exact-positive-integer? real? real? (real-in 0 1) (>=/c 0)))

(define/contract-out (percent x)
  (-> (real-in 0 1) string?)
  (real->decimal-string (* 100 x) 1))

(define/contract-out (epoch-row->strings row) ;; noqa
  (-> epoch-row/c (listof string?))
  (match-define (list epoch train-loss val-loss acc seconds) row)
  (list (~a epoch) (real->decimal-string train-loss 4) (real->decimal-string val-loss 4)
        (string-append (percent acc) "%") (real->decimal-string seconds 1)))

(module+ test
  (require rackunit
           (only-in (submod "..") [epoch-row->strings row->strings] [percent checked-percent]))
  (check-equal? (epoch-row->strings '(3 0.31234 0.29876 0.9123 12.345))
                '("3" "0.3123" "0.2988" "91.2%" "12.3"))
  (for ([row (in-list '((3 0.3 0.2 0.9) (0 0.3 0.2 0.9 1.0) (3 0.3 0.2 91.2 1.0)))])
    (check-exn #rx"epoch-row->strings: contract violation.*blaming: .*epoch-rows.rkt test[)]"
               (lambda () (row->strings row))))
  (check-exn #rx"percent: contract violation.*blaming: .*epoch-rows.rkt test[)]"
             (lambda () (checked-percent 1.5))))
