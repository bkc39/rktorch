#lang racket/base

(require racket/contract/base)

(provide
 (contract-out
  [candidates refactoring-suite?]))

(require bkc-style
         resyntax/base)

(define-refactoring-suite candidates
  #:suites (bkc-style))
