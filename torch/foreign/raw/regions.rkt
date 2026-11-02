#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe/atomic call-as-uninterruptible)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide with-cleanup
         with-uninterruptible)

(define-syntax-parse-rule (with-uninterruptible body:expr ...+)
  (call-as-uninterruptible (lambda () body ...)))

(define-syntax-parse-rule (with-cleanup body:expr ...+ #:finally cleanup:expr ...+)
  (dynamic-wind void (lambda () body ...) (lambda () cleanup ...)))
