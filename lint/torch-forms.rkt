#lang racket/base

(require racket/contract/base)

(provide
 (contract-out
  [torch-forms refactoring-suite?]))

(require (only-in torch + - * / add div mul sub)
         resyntax/base
         syntax/parse)

(define-refactoring-rule add-to-plus
  #:description "`+` is torch's `add` on tensors; use the operator."
  #:literals (add)
  (add a b)
  (+ a b))

(define-refactoring-rule sub-to-minus
  #:description "`-` is torch's `sub` on tensors; use the operator."
  #:literals (sub)
  (sub a b)
  (- a b))

(define-refactoring-rule mul-to-times
  #:description "`*` is torch's `mul` on tensors; use the operator."
  #:literals (mul)
  (mul a b)
  (* a b))

(define-refactoring-rule div-to-slash
  #:description "`/` is torch's `div` on tensors; use the operator."
  #:literals (div)
  (div a b)
  (/ a b))

(define-refactoring-suite torch-forms
  #:rules (add-to-plus sub-to-minus mul-to-times div-to-slash))
