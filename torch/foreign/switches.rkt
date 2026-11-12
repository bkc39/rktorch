#lang racket/base

(require (for-syntax racket/base syntax/parse/pre)
         (only-in racket/contract/base -> ->*)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "error.rkt" check-ok)
         (only-in "raw/global.rkt"
                  tr-get-cudnn-benchmark/raw
                  tr-get-cudnn-deterministic/raw
                  tr-get-deterministic-algorithms-warn-only/raw
                  tr-get-deterministic-algorithms/raw
                  tr-set-cudnn-benchmark/raw
                  tr-set-cudnn-deterministic/raw
                  tr-set-deterministic-algorithms/raw))

(define (read-flag raw who)
  (define-values (rc on?) (raw))
  (check-ok rc who)
  on?)

(define-syntax-parse-rule (define-flag getter:id raw-get:id
                            (~optional (~seq setter:id raw-set:id)))
  (begin
    (define/contract-out (getter) (-> boolean?) (read-flag raw-get 'getter))
    (~? (define/contract-out (setter on?)
          (-> boolean? void?)
          (check-ok (raw-set on?) 'setter)))))

(define/contract-out (use-deterministic-algorithms! on? ;; noqa
                                                    #:warn-only? [warn-only? #f])
  (->* [boolean?] [#:warn-only? boolean?] void?)
  (check-ok (tr-set-deterministic-algorithms/raw on? warn-only?)
            'use-deterministic-algorithms!))

(define-flag deterministic-algorithms-enabled? ;; noqa
  tr-get-deterministic-algorithms/raw)

(define-flag deterministic-algorithms-warn-only? ;; noqa
  tr-get-deterministic-algorithms-warn-only/raw)

(define-flag cudnn-deterministic? tr-get-cudnn-deterministic/raw ;; noqa
  set-cudnn-deterministic! tr-set-cudnn-deterministic/raw)

(define-flag cudnn-benchmark? tr-get-cudnn-benchmark/raw ;; noqa
  set-cudnn-benchmark! tr-set-cudnn-benchmark/raw)
