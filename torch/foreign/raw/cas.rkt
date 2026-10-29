#lang racket/base

(provide cas-add!
         cas-claim!
         cas-update!)

;; A failed `box-cas!` may be spurious, so each loop retries until the
;; update lands or, for a claim, until the box no longer holds `from`.
(define (cas-update! b f)
  (let retry ()
    (define old (unbox b))
    (define new (f old))
    (if (box-cas! b old new)
        new
        (retry))))

(define (cas-add! b n)
  (cas-update! b (lambda (old) (+ old n))))

(define (cas-claim! b from to)
  (let retry ()
    (cond
      [(box-cas! b from to) #t]
      [(eq? (unbox b) from) (retry)]
      [else #f])))
