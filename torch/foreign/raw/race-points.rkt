#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide race-hook
         race-point)

(define race-hook (box #f))

(define-syntax-parse-rule (race-point name:id subject:expr)
  (let ([hook (unbox race-hook)])
    (when hook
      (hook 'name subject))))
