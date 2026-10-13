#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/file display-lines-to-file)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/runtime-path define-runtime-path)
         (only-in threading ~>))

(define-runtime-path names-file
  (build-path 'up "codegen" "racket-names.txt"))

(define (export-names module)
  (module-declared? module #t)
  (define-values (variables syntaxes) (module->exports module))
  (for*/list ([phase+exports (in-list (append variables syntaxes))]
              [export (in-list (cdr phase+exports))])
    (symbol->string (car export))))

(module+ main
  (~> (append-map export-names '(racket threading))
      remove-duplicates
      (sort string<?)
      (display-lines-to-file names-file #:exists 'truncate)))
