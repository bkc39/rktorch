#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/unsafe/ops unsafe-struct*-cas! unsafe-struct*-ref)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide cas-add!
         cas-claim!
         cas-claim-when!
         cas-field-claim!
         cas-sites
         cas-take!
         cas-update!)

(module+ private
  (provide site-cas))

(begin-for-syntax
  (define known-sites
    '(update claim collector lifetime probe gate lock tensor-state))

  (define-syntax-class site
    #:description "a CAS site named in raw/cas.rkt"
    (pattern ((~literal quote) name:id)
             #:fail-unless (memq (syntax-e #'name) known-sites)
             (format "not one of ~s" known-sites))))

(define-syntax (site-list _stx)
  #`'#,known-sites)

(define cas-sites (site-list))

(struct slot (owner index))

(define (primitive-cas _site target old new)
  (if (box? target)
      (box-cas! target old new)
      (unsafe-struct*-cas! (slot-owner target) (slot-index target) old new)))

(define site-cas (box primitive-cas))

(define (cas! site target old new)
  ((unbox site-cas) site target old new))

(define (update-loop site b f)
  (let retry ()
    (define old (unbox b))
    (define new (f old))
    (if (cas! site b old new)
        new
        (retry))))

(define (take-loop site b)
  (let retry ()
    (define old (unbox b))
    (if (cas! site b old 0)
        old
        (retry))))

(define (claim-loop site target read claimable? to)
  (let retry ()
    (define held (read target))
    (cond
      [(not (claimable? held)) #f]
      [(cas! site target held to) #t]
      [else (retry)])))

(define (slot-ref s)
  (unsafe-struct*-ref (slot-owner s) (slot-index s)))

(define ((holding from) v)
  (eq? v from))

(define-syntax-parse-rule (cas-update! b:expr f:expr
                                       (~optional (~seq #:site s:site)
                                                  #:defaults ([s #''update])))
  (update-loop s b f))

(define-syntax-parse-rule (cas-add! b:expr n:expr)
  (let ([by n])
    (update-loop 'update b (lambda (old) (+ old by)))))

(define-syntax-parse-rule (cas-take! b:expr)
  (take-loop 'update b))

(define-syntax-parse-rule (cas-claim! b:expr from:expr to:expr
                                      (~optional (~seq #:site s:site)
                                                 #:defaults ([s #''claim])))
  (claim-loop s b unbox (holding from) to))

(define-syntax-parse-rule (cas-claim-when! b:expr claimable?:expr to:expr #:site s:site)
  (claim-loop s b unbox claimable? to))

(define-syntax-parse-rule (cas-field-claim! owner:expr index:expr from:expr to:expr
                                            #:site s:site)
  (claim-loop s (slot owner index) slot-ref (holding from) to))
