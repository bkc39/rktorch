#lang racket/base

(require (only-in racket/file make-directory* make-temporary-directory)
         (only-in (submod torch/foreign/raw/native-location checked) native-library-directory)
         (only-in "../harness.rkt" with-temporary-path)
         (only-in "../shim.rkt" torch-dir))

(provide call-with-pre-hook-torch)

(define (write-module! dir name form)
  (make-directory* dir)
  (call-with-output-file (build-path dir (format "~a.rkt" name))
    (lambda (port) (write `(module ,name racket/base ,@form) port))))

(define (write-reexport! fake name)
  (define real `(file ,(path->string (build-path torch-dir (format "~a.rkt" name)))))
  (write-module! fake name `((require ,real) (provide (all-from-out ,real)))))

(define (write-absent! absent name)
  (write-module! (build-path absent "torch" "foreign" "raw") name
                 `((error ',name "is not in a checkout this old"))))

(define (call-with-pre-hook-torch proc)
  (with-temporary-path (collects (make-temporary-directory "collects-~a"))
    (define fake (build-path collects "old" "torch"))
    (define absent (build-path collects "absent"))
    (for-each (lambda (name) (write-reexport! fake name)) '(main nn))
    (make-file-or-directory-link (native-library-directory) (build-path fake "native-libs"))
    (for-each (lambda (name) (write-absent! absent name)) '(native-location pressure-settings))
    (proc (cons "PLTCOLLECTS"
                (format "~a:~a:" (path->string (build-path collects "old"))
                        (path->string absent))))))
