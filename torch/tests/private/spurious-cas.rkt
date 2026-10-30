#lang racket/base

(require (only-in racket/contract/base -> any listof or/c)
         (only-in "../../private/contract.rkt" define/contract-out)
         (only-in "../../foreign/raw/cas.rkt" cas-sites)
         (only-in (submod "../../foreign/raw/cas.rkt" private) site-cas)
         (only-in "../../foreign/raw/regions.rkt" with-cleanup))

(define cas-site/c (apply or/c cas-sites))

;; The first attempt at each named site fails with the target unchanged. The
;; bookkeeping calls the real primitive: through the seam it would recurse.
(define/contract-out (call-with-spurious-cas-failures sites thunk)
  (-> (listof cas-site/c) (-> any) any)
  (define real (unbox site-cas))
  (define failed (box '()))
  (define (first-at? site)
    (let retry ()
      (define seen (unbox failed))
      (cond
        [(or (not (memq site sites)) (memq site seen)) #f]
        [(real 'update failed seen (cons site seen)) #t]
        [else (retry)])))
  (define (spurious site target old new)
    (and (not (first-at? site)) (real site target old new)))
  (set-box! site-cas spurious)
  (with-cleanup (thunk) #:finally (set-box! site-cas real)))
