#lang racket/base

(provide callback?
         leg-message
         not-implemented
         prop:callback
         prop:trainer
         scheduler-config?
         trainer-automatic?
         trainer-defines?
         trainer-hparam-names
         trainer?
         (struct-out trainer-info))

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

(define (leg-message leg)
  (format "not implemented until ~a" (hash-ref legs leg)))

(define (not-implemented who leg)
  (raise (exn:fail:unsupported (format "~a: ~a" who (leg-message leg))
                               (current-continuation-marks))))
