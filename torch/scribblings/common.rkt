#lang at-exp racket/base

;; The manual's shared preamble, after racket-doc's reference/mz.rkt.
;;
;; `torch-examples` binds every chapter's examples to one evaluator, so the
;; results in the manual are produced by the library at build time rather
;; than pasted in beside it.
;;
;; The `for-label` imports are deliberately NOT re-exported here yet.  A
;; binding re-exported through this module is tagged to this module, while
;; the reference chapters import `(only-in torch ...)` directly and tag to
;; `torch`; the two would not match, so a guide link would miss an entry
;; that does exist.  Chapters carry their own `for-label` line until the
;; reference moves onto this module too, and then both sides can move
;; together.

(require scribble/example
         scribble/manual)

(provide (all-from-out scribble/example)
         (all-from-out scribble/manual)
         torch-eval
         torch-examples)

(define torch-eval (make-base-eval '(require torch torch/nn)))

(define-syntax-rule (torch-examples body ...)
  (examples #:eval torch-eval body ...))
