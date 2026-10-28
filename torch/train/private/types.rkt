#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre expr id))
         (only-in racket/contract/base contract? flat-named-contract)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../nn/layer.rkt" layer?)
         (only-in (submod "../../nn/layer.rkt" private) layer-method)
         (only-in "../../private/contract.rkt" define/contract-out))

(provide callback-hook
         callback-kind
         callback-state-key
         callback-type
         callback?
         define-named-contract
         method-hook
         method-state-key
         not-implemented
         prop:callback
         prop:trainer
         (struct-out scheduler-config)
         trainer-automatic?
         trainer-defines?
         trainer-hook
         trainer-hparam-names
         trainer?
         (struct-out callback-info)
         (struct-out trainer-info))

(define-syntax-parse-rule (define-named-contract name:id contract:expr)
  (define/contract-out name contract? (flat-named-contract 'name contract)))

(struct trainer-info (forms automatic? hparams))

(struct callback-info (kind type hook-ref state-key))

(define-values (prop:trainer trainer? trainer-info-of)
  (make-struct-type-property 'trainer))

(define-values (prop:callback callback? callback-info-of)
  (make-struct-type-property 'callback))

(struct scheduler-config (scheduler interval frequency))

(define (trainer-defines? tr form)
  (and (memq form (trainer-info-forms (trainer-info-of tr))) #t))

(define (trainer-automatic? tr)
  (trainer-info-automatic? (trainer-info-of tr)))

(define (trainer-hparam-names tr)
  (trainer-info-hparams (trainer-info-of tr)))

(define (trainer-hook tr name)
  (and (layer? tr) (method-hook tr name)))

(define (method-hook layer name)
  (layer-method layer name))

(define (method-state-key cb)
  ((layer-method cb 'state-key) cb))

(define (callback-hook cb name)
  ((callback-info-hook-ref (callback-info-of cb)) cb name))

(define (callback-kind cb)
  (callback-info-kind (callback-info-of cb)))

(define (callback-type cb)
  (callback-info-type (callback-info-of cb)))

(define (callback-state-key cb)
  ((callback-info-state-key (callback-info-of cb)) cb))

(define legs #hasheq((T1 . "T1 (#384)") (T2 . "T2 (#385)") (T3 . "T3 (#386)")))

(define (not-implemented who leg)
  (raise (exn:fail:unsupported
          (format "~a: not implemented until ~a" who (hash-ref legs leg))
          (current-continuation-marks))))
