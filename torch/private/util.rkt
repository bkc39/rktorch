#lang racket/base

;; Small filesystem helpers shared across the package.

(require (only-in racket/file make-temporary-file))

(provide call-with-temporary-file
         with-temporary-file)

;; Deletes the temp file on any escape; the file-exists? guard lets `proc`
;; rename it away. Creation sits immediately before the dynamic-wind, so no
;; user code can run between them and leak the file.
(define (call-with-temporary-file proc
                                  #:template [template "rktorch-~a.tmp"]
                                  #:directory [directory #f])
  (define tmp (make-temporary-file template #f directory))
  (dynamic-wind
   void
   (lambda () (proc tmp))
   (lambda () (when (file-exists? tmp) (delete-file tmp)))))

(define-syntax-rule (with-temporary-file (path-id kw-arg ...) body ...)
  (call-with-temporary-file (lambda (path-id) body ...) kw-arg ...))
