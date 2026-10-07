#lang racket/base

(require (for-syntax racket/base)
         ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define)

(provide race-hook
         race-point)

(define race-hook (box #f))

(define-syntax-parse-rule (race-point name:id subject:expr)
  (let ([hook (unbox race-hook)])
    (when hook
      (hook 'name subject))))
