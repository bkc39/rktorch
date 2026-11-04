#lang racket/base

(provide ladder
         side-layers
         layer-pieces
         op-cost-names)

(define ladder '(raw registered retry accounted checked wrapped unchecked facade))

(define side-layers '(contracted))

(define layer-pieces
  (hasheq 'raw "op (libtorch, call and free crossings)"
          'registered "finalizer registration and finalization"
          'retry "oom-retry wrapper"
          'accounted "accounting (placement, record-allocation!)"
          'checked "pressure check"
          'wrapped "shape readback and tensor struct"
          'unchecked "dispatch"
          'facade "facade (operator or contract)"
          'contracted "contract (contracted add over dispatch)"))

(define op-cost-names '(scalar-add add-8x8 zeros-8x8 narrow stack-32 cifar-chain))
