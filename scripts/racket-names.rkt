#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/runtime-path define-runtime-path))

(define-runtime-path names-file
  (build-path 'up "codegen" "racket-names.txt"))

(module+ main
  (require (only-in racket/file display-lines-to-file)
           (only-in racket/list append-map remove-duplicates)
           (only-in threading ~>)
           (only-in "../torch/private/exports.rkt" export-names))
  (~> (append-map export-names '(racket threading))
      remove-duplicates
      (sort symbol<?)
      (map symbol->string _)
      (display-lines-to-file names-file #:exists 'truncate)))
