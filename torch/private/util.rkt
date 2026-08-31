#lang racket/base

(require (only-in racket/file
                  delete-directory/files make-temporary-file)
         ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define)

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

(define-syntax-parse-rule
  (with-temporary-file (path-id:id kw-arg ...) body:expr ...+)
  (call-with-temporary-file (lambda (path-id) body ...) kw-arg ...))

(define (call-with-temporary-directory proc
                                       #:template [template "rktorch-~a.tmp"]
                                       #:directory [directory #f])
  (define dir (make-temporary-file template 'directory directory))
  (dynamic-wind
   void
   (lambda () (proc dir))
   (lambda () (delete-directory/files dir #:must-exist? #f))))

(define-syntax-parse-rule
  (with-temporary-directory (dir-id:id kw-arg ...) body:expr ...+)
  (call-with-temporary-directory (lambda (dir-id) body ...) kw-arg ...))
