#lang racket/base

(provide cas!
         cas-add!
         cas-claim!
         cas-update!)

;; The test seam, as structs.rkt's `state-cas`: the primitive every site
;; calls, which a test swaps for one that fails as a spurious `box-cas!`
;; may, with the box unchanged.
(module+ private
  (provide site-cas))

(define site-cas (box (lambda (_site b old new) (box-cas! b old new))))

(define (cas! site b old new)
  ((unbox site-cas) site b old new))

;; A failed CAS may be spurious, so each loop retries until the update lands
;; or, for a claim, until the box no longer holds `from`.
(define (cas-update! b f)
  (let retry ()
    (define old (unbox b))
    (define new (f old))
    (if (cas! 'update b old new)
        new
        (retry))))

(define (cas-add! b n)
  (cas-update! b (lambda (old) (+ old n))))

(define (cas-claim! b from to)
  (let retry ()
    (cond
      [(cas! 'claim b from to) #t]
      [(eq? (unbox b) from) (retry)]
      [else #f])))
