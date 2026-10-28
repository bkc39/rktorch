#lang racket/base

(require (only-in racket/contract/base ->)
         (only-in racket/math nan?)
         (only-in threading ~>)
         (only-in "contract.rkt" define/contract-out))

;; The Maclaurin series of erf(q)/q in 160-bit fixed point, times q and
;; 2/sqrt(pi) to 41 digits, rounds once: the result is the correctly rounded
;; erf of the double it is given, short of a tie within 2^-140.
(define two/sqrt-pi #e1.1283791670955125738961589031215451716881)

(define precision 160)

(define (series q)
  (define q² (* q q))
  (define num (numerator q²))
  (define den (denominator q²))
  (define one (arithmetic-shift 1 precision))
  (let loop ([n 0] [power one] [acc 0])
    (cond
      [(zero? power) (* two/sqrt-pi q (/ acc one))]
      [else
       (define term (quotient power (add1 (* 2 n))))
       (loop (add1 n)
             (quotient (* power num) (* (add1 n) den))
             (if (even? n) (+ acc term) (- acc term)))])))

(define/contract-out (erf x) (-> flonum? flonum?)
  (cond
    [(or (nan? x) (zero? x)) x]
    [(>= x 6.0) 1.0]
    [(<= x -6.0) -1.0]
    [else (~> x inexact->exact series exact->inexact)]))
