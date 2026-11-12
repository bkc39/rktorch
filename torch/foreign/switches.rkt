#lang racket/base

(require (for-syntax racket/base syntax/parse/pre)
         (only-in racket/contract/base -> ->* or/c)
         (only-in racket/list index-of)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "error.rkt" check-ok)
         (only-in "raw/global.rkt"
                  tr-get-cuda-matmul-allow-tf32/raw
                  tr-get-cudnn-allow-tf32/raw
                  tr-get-cudnn-benchmark/raw
                  tr-get-cudnn-deterministic/raw
                  tr-get-deterministic-algorithms-warn-only/raw
                  tr-get-deterministic-algorithms/raw
                  tr-get-float32-matmul-precision/raw
                  tr-set-cuda-matmul-allow-tf32/raw
                  tr-set-cudnn-allow-tf32/raw
                  tr-set-cudnn-benchmark/raw
                  tr-set-cudnn-deterministic/raw
                  tr-set-deterministic-algorithms/raw
                  tr-set-float32-matmul-precision/raw))

(define (read-flag raw who)
  (define-values (rc on?) (raw))
  (check-ok rc who)
  on?)

(begin-for-syntax
  (define-splicing-syntax-class codec
    #:attributes (contract decode encode)
    (pattern (~seq #:among (choice:id ...+))
             #:with contract #'(or/c 'choice ...)
             #:with decode #'(lambda (code) (list-ref '(choice ...) code))
             #:with encode #'(lambda (value) (index-of '(choice ...) value)))
    (pattern (~seq)
             #:with contract #'boolean?
             #:with decode #'values
             #:with encode #'values)))

(define-syntax-parse-rule (define-flag getter:id raw-get:id
                            (~optional (~seq setter:id raw-set:id))
                            flag:codec)
  (begin
    (define/contract-out (getter)
      (-> flag.contract)
      (flag.decode (read-flag raw-get 'getter)))
    (~? (define/contract-out (setter value)
          (-> flag.contract void?)
          (check-ok (raw-set (flag.encode value)) 'setter)))))

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

(define-flag float32-matmul-precision tr-get-float32-matmul-precision/raw ;; noqa
  set-float32-matmul-precision! tr-set-float32-matmul-precision/raw
  #:among (highest high medium))

(define-flag cuda-matmul-allow-tf32? tr-get-cuda-matmul-allow-tf32/raw ;; noqa
  set-cuda-matmul-allow-tf32! tr-set-cuda-matmul-allow-tf32/raw)

(define-flag cudnn-allow-tf32? tr-get-cudnn-allow-tf32/raw ;; noqa
  set-cudnn-allow-tf32! tr-set-cudnn-allow-tf32/raw)
