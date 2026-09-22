#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/contract/base -> ->* any or/c)
         ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "device-type.rkt" device-type device?)
         (only-in "error.rkt" check-ok)
         (only-in "ops.rkt" default-device)
         (only-in "raw/autocast.rkt"
                  tr-autocast-dtype/raw
                  tr-is-autocast-enabled/raw
                  tr-set-autocast-enabled/raw))

(provide with-autocast)

(define half-dtype/c (or/c 'float16 'bfloat16))
(define device-spec/c (or/c 'cpu 'cuda 'mps device?))

(define (type-of spec)
  (if (device? spec) (device-type spec) spec))

(define/contract-out (autocast-enabled? [device (default-device)]) ;; noqa
  (->* [] [device-spec/c] boolean?)
  (define-values (rc on?) (tr-is-autocast-enabled/raw (type-of device)))
  (check-ok rc 'autocast-enabled?)
  on?)

(define/contract-out (autocast-dtype [device (default-device)]) ;; noqa
  (->* [] [device-spec/c] half-dtype/c)
  (define-values (rc dtype) (tr-autocast-dtype/raw (type-of device)))
  (check-ok rc 'autocast-dtype)
  dtype)

(define (set-autocast! type dtype on?)
  (check-ok (tr-set-autocast-enabled/raw type dtype on?)
            'set-autocast!))

(define/contract-out (call-with-autocast thunk ;; noqa
                                         #:device [device (default-device)]
                                         #:dtype [dtype 'bfloat16])
  (->* [(-> any)] [#:device device-spec/c #:dtype half-dtype/c] any)
  ;; the with-autocast form expands inside this module, past the boundary
  ;; contract, so the arguments are judged here as well
  (unless (device-spec/c device)
    (raise-argument-error 'with-autocast "(or/c 'cpu 'cuda 'mps device?)"
                          device))
  (unless (half-dtype/c dtype)
    (raise-argument-error 'with-autocast "(or/c 'float16 'bfloat16)" dtype))
  (define type (type-of device))
  (define was-on? (autocast-enabled? type))
  (define was-dtype (autocast-dtype type))
  (dynamic-wind (lambda () (set-autocast! type dtype #t))
                thunk
                (lambda () (set-autocast! type was-dtype was-on?))))

(define-syntax-parse-rule (with-autocast
                            (~alt (~optional (~seq #:device device:expr)
                                             #:defaults ([device #'(default-device)]))
                                  (~optional (~seq #:dtype dtype:expr)
                                             #:defaults ([dtype #''bfloat16])))
                            ...
                            body:expr ...+)
  (call-with-autocast (lambda () body ...) #:device device #:dtype dtype))
