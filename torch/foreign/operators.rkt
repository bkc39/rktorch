#lang racket/base

;; Arithmetic operators that shadow racket/base: a numeric fast path falls
;; through untouched, any tensor operand routes into libtorch, and chains
;; fold left. The t+/t-/t*/t/ spellings keep this module's own arithmetic
;; racket/base; foreign.rkt renames them on the way out. `@` is matmul.

(require (only-in racket/base
                  [+ base:+]
                  [- base:-]
                  [* base:*]
                  [/ base:/])
         (only-in "raw/syntax.rkt" define-arith)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" add div matmul mul neg sub))

(provide t+ t- t* t/ @)

;; Unary forms mirror racket/base: (- t) negates, (/ t) is the reciprocal,
;; (+ t) and (* t) are the identity.
(define-arith t+ tensor? add base:+ values)
(define-arith t- tensor? sub base:- neg)
(define-arith t* tensor? mul base:* values)
(define-arith t/ tensor? div base:/ (lambda (t) (div 1.0 t)))

;; Matmul chains left like Python: (@ a b c) is ((a @ b) @ c).
(define (@ a . rest)
  (foldl (lambda (b acc) (matmul acc b)) a rest))
