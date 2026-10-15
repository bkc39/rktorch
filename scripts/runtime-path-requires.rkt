#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/list append-map)
         (only-in racket/match match)
         (only-in racket/path file-name-from-path)
         (only-in racket/port port->list)
         (only-in racket/runtime-path define-runtime-path)
         (only-in racket/string string-prefix?)
         (only-in syntax/modread with-module-reading-parameterization))

(define-runtime-path repository-root "..")

(define phase-shifts '(for-syntax for-template for-label for-meta for-space))

(define (whole-specs spec)
  (match (syntax-e spec)
    ['racket/runtime-path (list spec)]
    [(cons head _)
     #:when (memq (syntax-e head) phase-shifts)
     (append-map whole-specs (cdr (syntax->list spec)))]
    [_ '()]))

(define (whole-runtime-path-specs stx)
  (match (syntax->list stx)
    [(cons head _)
     #:when (memq (syntax-e head) '(quote quasiquote syntax quasisyntax))
     '()]
    [(cons head specs)
     #:when (eq? (syntax-e head) 'require)
     (append-map whole-specs specs)]
    [(? list? forms) (append-map whole-runtime-path-specs forms)]
    [#f '()]))

(define (read-forms path)
  (call-with-input-file path
    (lambda (in)
      (port-count-lines! in)
      (with-module-reading-parameterization
       (lambda ()
         (port->list (lambda (port) (read-syntax path port)) in))))))

(define (skipped-directory? path)
  (define name (path->string (file-name-from-path path)))
  (or (string-prefix? name ".")
      (member name '("compiled" "manual" "result"))))

(define (repository-offenders)
  (define (descend? dir) (not (skipped-directory? dir)))
  (for*/list ([path (in-directory repository-root descend?)]
              #:when (regexp-match? #rx"[.]rkt$" (path->string path))
              [spec (in-list (append-map whole-runtime-path-specs
                                         (read-forms path)))])
    (format "~a:~a" (simplify-path path) (syntax-line spec))))

(module+ test
  (require (only-in rackunit check-equal? test-case))

  (define (offenders datum)
    (map syntax->datum (whole-runtime-path-specs (datum->syntax #f datum))))

  (test-case "a whole racket/runtime-path is found"
    (check-equal? (offenders '(require racket/runtime-path))
                  '(racket/runtime-path)))

  (test-case "a whole require in a submodule or a phase shift is found"
    (check-equal? (offenders '(module+ test
                                (require rackunit racket/runtime-path)))
                  '(racket/runtime-path))
    (check-equal? (offenders '(require (for-syntax racket/runtime-path)))
                  '(racket/runtime-path)))

  (test-case "a whole require behind a comment is found"
    (define source "(require ;; on purpose\n racket/runtime-path)")
    (check-equal?
     (map syntax->datum
          (whole-runtime-path-specs
           (read-syntax 'source (open-input-string source))))
     '(racket/runtime-path)))

  (test-case "a quoted require is data"
    (check-equal? (offenders ''(require racket/runtime-path)) '()))

  (test-case "only-in racket/runtime-path passes"
    (check-equal? (offenders '(require (for-syntax racket/base)
                                       (only-in racket/runtime-path
                                                define-runtime-path)))
                  '()))

  (test-case "no .rkt file in the repository requires racket/runtime-path whole"
    (check-equal? (repository-offenders) '())))
