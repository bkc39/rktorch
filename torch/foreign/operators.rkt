#lang racket/base

;; Arithmetic operators that shadow racket/base, in the rkt-polars style
;; (polars/private/generic/operators.rkt): a numeric fast path falls through
;; to racket/base untouched, any tensor operand routes into libtorch with
;; the usual broadcasting, and longer chains fold left. The shadowing public
;; names live behind t+/t-/t*/t/ spellings so this module's own arithmetic
;; stays racket/base; foreign.rkt renames them on the way out.
;;
;; `@` is the matmul operator (Python's a @ b); it has no racket/base
;; collision, so it is exported under its own name.

(require (only-in racket/base
                  [+ base:+]
                  [- base:-]
                  [* base:*]
                  [/ base:/])
         (only-in "raw/syntax.rkt" define-arith)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" add div matmul mul neg sub))

(provide t+ t- t* t/ @)

;; Unary forms mirror racket/base: (- t) negates, (/ t) is the reciprocal;
;; (+ t) and (* t) are the identity, like (+ 5). define-arith (raw/syntax.rkt)
;; takes the predicate and ops as arguments, so dispatch resolves here.
(define-arith t+ tensor? add base:+ values)
(define-arith t- tensor? sub base:- neg)
(define-arith t* tensor? mul base:* values)
(define-arith t/ tensor? div base:/ (lambda (t) (div 1.0 t)))

;; Matmul chains left like Python: (@ a b c) is ((a @ b) @ c).
(define (@ a . rest)
  (foldl (lambda (b acc) (matmul acc b)) a rest))
