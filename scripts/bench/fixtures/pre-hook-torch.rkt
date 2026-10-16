#lang racket/base

(require (only-in racket/file make-directory* make-temporary-directory)
         (only-in "../harness.rkt" with-temporary-path)
         (only-in "../shim.rkt" torch-dir))

(provide call-with-pre-hook-torch)

(define (write-reexport! fake name)
  (define real `(file ,(path->string (build-path torch-dir (format "~a.rkt" name)))))
  (call-with-output-file (build-path fake (format "~a.rkt" name))
    (lambda (port)
      (write `(module ,name racket/base (require ,real) (provide (all-from-out ,real))) port))))

(define (call-with-pre-hook-torch proc)
  (with-temporary-path (collects (make-temporary-directory "collects-~a"))
    (define fake (build-path collects "torch"))
    (make-directory* fake)
    (for-each (lambda (name) (write-reexport! fake name)) '(main nn))
    (make-file-or-directory-link (build-path torch-dir "native-libs")
                                 (build-path fake "native-libs"))
    (proc (cons "PLTCOLLECTS" (format "~a:" (path->string collects))))))
