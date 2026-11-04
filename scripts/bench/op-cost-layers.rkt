#lang racket/base

(provide ladder
         side-layers
         layer-pieces
         leave-one-out
         loo-base
         pure-overhead-ops)

(define ladder '(raw registered retry accounted checked wrapped unchecked facade))

(define side-layers '(contracted without-retry without-accounting without-pressure))

;; a ladder layer and the `loo-base` stack with that layer's piece taken out
(define leave-one-out
  (hasheq 'retry 'without-retry 'accounted 'without-accounting 'checked 'without-pressure))

(define loo-base 'wrapped)

(define pure-overhead-ops '(scalar-add add-8x8))

(define layer-pieces
  (hasheq 'raw "op (libtorch, call and free crossings)"
          'registered "finalizer registration and finalization"
          'retry "oom-retry wrapper"
          'accounted "accounting (account!)"
          'checked "pressure check"
          'wrapped "shape readback and tensor struct"
          'unchecked "dispatch"
          'facade "facade (operator or contract)"
          'contracted "contract (contracted add over dispatch)"))
