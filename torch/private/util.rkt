#lang racket/base

(require (only-in racket/file make-temporary-file))

(provide call-with-temporary-file
         with-temporary-file)

;; deletes the temp file on any escape; the file-exists? guard lets `proc`
;; rename it away
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
