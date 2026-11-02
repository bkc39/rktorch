#lang racket/base

(module+ test
  (require (only-in racket/file file->string find-files)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit)

  (define-runtime-path tests-dir ".")

  (define relative-example-rx #px"\"(?:\\.\\./)+scribblings/examples/")

  (test-case "no source-tree test requires a literate example by relative path"
    (for ([file (in-list (find-files (lambda (p) (regexp-match? #rx"\\.rkt$" p)) tests-dir))])
      (check-false (regexp-match? relative-example-rx (file->string file))
                   (format "~a: require the example by collection, from examples/test" file)))))
