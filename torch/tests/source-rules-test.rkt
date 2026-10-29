#lang racket/base

(module+ test
  (require (only-in racket/file file->lines find-files)
           (only-in racket/list append-map)
           (only-in racket/match match)
           (only-in racket/path normalize-path)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit)

  (define-runtime-path torch-dir "..")
  (define-runtime-path foreign-dir "../foreign")
  (define-runtime-path cas-module "../foreign/raw/cas.rkt")

  (define (modules-under dir)
    (find-files (lambda (p) (regexp-match? #rx"[.]rkt$" (path->string p))) dir))

  (define (module-forms path)
    (parameterize ([read-accept-reader #t] [read-accept-lang #t])
      (call-with-input-file path read)))

  (define (atoms v)
    (match v
      [(cons a d) (append (atoms a) (atoms d))]
      [(? vector?) (atoms (vector->list v))]
      [(? box?) (atoms (unbox v))]
      [_ (list v)]))

  (define (symbols-matching rx path)
    (for/list ([a (in-list (atoms (module-forms path)))]
               #:when (and (symbol? a) (regexp-match? rx (symbol->string a))))
      a))

  (define (check-no-symbols rx paths what)
    (for ([path (in-list paths)])
      (check-equal? (symbols-matching rx path) '() (format "~a ~a" path what))))

  (define (same-file? a b)
    (equal? (normalize-path a) (normalize-path b)))

  (define foreign-modules (modules-under foreign-dir))

  (test-case "the scans read the checkout under test"
    (check-true (> (length foreign-modules) 20))
    (check-true (for/or ([p (in-list foreign-modules)]) (same-file? p cas-module))))

  (test-case "no module under torch/foreign enters atomic mode"
    (check-no-symbols
     #px"(^|:)ffi/unsafe/alloc$|(call-as-atomic|start-atomic|end-atomic|start-breakable-atomic|end-breakable-atomic)$"
     foreign-modules
     "enters atomic mode, or names a form that does"))

  (test-case "only raw/cas.rkt calls a CAS primitive"
    (check-no-symbols
     #px"(box-cas!|box[*]-cas!|struct-cas!|struct[*]-cas!)$"
     (for/list ([p (in-list (modules-under torch-dir))]
                #:unless (same-file? p cas-module))
       p)
     "calls a CAS primitive instead of raw/cas.rkt's loops"))

  (define (issue-comments path)
    (for/list ([line (in-list (file->lines path))]
               #:when (regexp-match? #px";[^\"]*#[0-9]+" line))
      line))

  (test-case "comments under torch/foreign cite no issue numbers"
    (check-equal? (append-map issue-comments foreign-modules) '())))
