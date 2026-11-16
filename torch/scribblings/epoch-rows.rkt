#lang racket/base

(require (only-in racket/format ~a)
         (only-in racket/match match-define))

(provide epoch-row->strings
         percent)

(define (percent x)
  (real->decimal-string (* 100 x) 1))

(define (epoch-row->strings row)
  (match-define (list epoch train-loss val-loss acc seconds) row)
  (list (~a epoch) (real->decimal-string train-loss 4) (real->decimal-string val-loss 4)
        (string-append (percent acc) "%") (real->decimal-string seconds 1)))

(module+ test
  (require rackunit)
  (check-equal? (epoch-row->strings '(3 0.31234 0.29876 0.9123 12.345))
                '("3" "0.3123" "0.2988" "91.2%" "12.3")))
