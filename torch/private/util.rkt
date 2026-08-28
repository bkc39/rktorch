#lang racket/base

(require (only-in racket/file
                  delete-directory/files make-temporary-directory
                  make-temporary-file))

(provide call-with-temporary-directory
         call-with-temporary-file
         with-temporary-directory
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

(define (call-with-temporary-directory proc)
  (define dir (make-temporary-directory))
  (dynamic-wind
   void
   (lambda () (proc dir))
   (lambda () (delete-directory/files dir #:must-exist? #f))))

(define-syntax-rule (with-temporary-directory (dir-id) body ...)
  (call-with-temporary-directory (lambda (dir-id) body ...)))
