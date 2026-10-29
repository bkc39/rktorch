#lang racket/base

(module+ test
  (require (only-in racket/file find-files)
           (only-in racket/match match)
           rackunit)

  ;; A parallel thread's atomic section runs on main's OS thread, so no
  ;; binding may enter one (#296).
  (define forbidden '(ffi/unsafe/alloc call-as-atomic start-atomic))

  (define (module-forms path)
    (parameterize ([read-accept-reader #t] [read-accept-lang #t])
      (call-with-input-file path read)))

  (define (atoms v)
    (match v
      [(cons a d) (append (atoms a) (atoms d))]
      [_ (list v)]))

  (define foreign-modules
    (find-files (lambda (p) (regexp-match? #rx"[.]rkt$" (path->string p)))
                (collection-file-path "foreign" "torch")))

  (test-case "no module under torch/foreign enters atomic mode"
    (check-true (> (length foreign-modules) 20))
    (for* ([path (in-list foreign-modules)]
           [name (in-list forbidden)])
      (check-false (memq name (atoms (module-forms path)))
                   (format "~a names ~a" path name)))))
