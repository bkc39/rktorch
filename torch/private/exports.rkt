#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in threading ~>))

(provide export-names)

(define (export-names module [phase-ok? (lambda (_) #t)])
  (module-declared? module #t)
  (define-values (variables syntaxes) (module->exports module))
  (~> (for*/list ([phase+exports (in-list (append variables syntaxes))]
                  #:when (phase-ok? (car phase+exports))
                  [export (in-list (cdr phase+exports))])
        (car export))
      remove-duplicates
      (sort symbol<?)))
