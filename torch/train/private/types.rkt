#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre expr id))
         (only-in racket/contract/base contract? flat-named-contract)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../private/contract.rkt" define/contract-out))

(provide callback?
         define-named-contract
         not-implemented
         prop:callback
         prop:trainer
         scheduler-config?
         trainer-automatic?
         trainer-defines?
         trainer-hparam-names
         trainer?
         (struct-out trainer-info))

(define-syntax-parse-rule (define-named-contract name:id contract:expr)
  (define/contract-out name contract? (flat-named-contract 'name contract)))

(struct trainer-info (forms automatic? hparams))

(define-values (prop:trainer trainer? trainer-info-of)
  (make-struct-type-property 'trainer))

(define-values (prop:callback callback? callback-info-of) ;; noqa
  (make-struct-type-property 'callback))

(struct scheduler-config (scheduler interval frequency))

(define (trainer-defines? tr form)
  (and (memq form (trainer-info-forms (trainer-info-of tr))) #t))

(define (trainer-automatic? tr)
  (trainer-info-automatic? (trainer-info-of tr)))

(define (trainer-hparam-names tr)
  (trainer-info-hparams (trainer-info-of tr)))

(define legs #hasheq((T1 . "T1 (#384)") (T2 . "T2 (#385)") (T3 . "T3 (#386)")))

(define (not-implemented who leg)
  (raise (exn:fail:unsupported
          (format "~a: not implemented until ~a" who (hash-ref legs leg))
          (current-continuation-marks))))
