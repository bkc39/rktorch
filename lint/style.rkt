#lang racket/base

(require racket/contract/base)

(provide
 (contract-out
  [project-style refactoring-suite?]))

(require resyntax/base
         rktorch-lint/torch-forms
         resyntax/default-recommendations)

(define-refactoring-suite project-style
  #:suites (default-recommendations
            torch-forms))
