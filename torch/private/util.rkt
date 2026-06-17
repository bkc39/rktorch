#lang racket/base

;; Small filesystem helpers shared across the package. call-with-temporary-file
;; owns a scratch file's whole lifetime so callers don't hand-roll the
;; create/dynamic-wind/delete dance (and can't strand the file on an error).

(require (only-in racket/file make-temporary-file))

(provide call-with-temporary-file
         with-temporary-file)

;; Create a temp file (named from `template`, in `directory` or the system temp
;; dir), call `proc` with its path, and delete it on the way out — even on
;; escape, and even if `proc` has renamed it away (the delete is guarded by
;; file-exists?). The file is created immediately before the dynamic-wind, so the
;; only code between creation and the cleanup-registering wind is this function's
;; own frame: nothing user-supplied can run there and leak the file.
(define (call-with-temporary-file proc
                                  #:template [template "rktorch-~a.tmp"]
                                  #:directory [directory #f])
  (define tmp (make-temporary-file template #f directory))
  (dynamic-wind
   void
   (lambda () (proc tmp))
   (lambda () (when (file-exists? tmp) (delete-file tmp)))))

;; (with-temporary-file (path-id #:template t #:directory d) body ...) — bind the
;; temp file's path to path-id for the body; keyword args are optional.
(define-syntax-rule (with-temporary-file (path-id kw-arg ...) body ...)
  (call-with-temporary-file (lambda (path-id) body ...) kw-arg ...))
