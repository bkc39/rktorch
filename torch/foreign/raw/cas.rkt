#lang racket/base

(provide call-with-spurious-cas-failures
         cas!
         cas-add!
         cas-claim!
         cas-update!)

;; The test seam: while a test injects, the first attempt at each named site
;; fails as a spurious `box-cas!` failure may, with the box unchanged.
(define injector (box #f))

(define (cas! site b old new)
  (define inject (unbox injector))
  (and (not (and inject (inject site)))
       (box-cas! b old new)))

(define (call-with-spurious-cas-failures sites thunk)
  (define failed (box '()))
  (define (inject site)
    (let retry ()
      (define seen (unbox failed))
      (cond
        [(or (not (memq site sites)) (memq site seen)) #f]
        [(box-cas! failed seen (cons site seen)) #t]
        [else (retry)])))
  (dynamic-wind (lambda () (set-box! injector inject))
                thunk
                (lambda () (set-box! injector #f))))

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
