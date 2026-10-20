#lang racket/base

(module+ test
  (require (only-in compiler/find-exe find-exe)
           (only-in racket/list filter-map)
           (only-in racket/match match)
           (only-in racket/port port->lines)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*)
           rackunit)

  (define-runtime-path collection "..")
  (define-runtime-path smoke "../pkg-build/smoke.rkt")

  (define (environment #:service? service?)
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"PLT_PKG_BUILD_SERVICE" (and service? #"1"))
    (environment-variables-set! env #"PLT_INFO_ALLOW_VARS" (and service? #"PLT_PKG_BUILD_SERVICE"))
    env)

  (define (racket-lines env . args)
    (define out (open-output-string))
    (define err (open-output-string))
    (define ok?
      (parameterize ([current-environment-variables env]
                     [current-output-port out]
                     [current-error-port err])
        (apply system* (find-exe) args)))
    (check-equal? (get-output-string err) "")
    (check-true ok?)
    (port->lines (open-input-string (get-output-string out))))

  (define (module-run line)
    (match (regexp-match #rx"^raco test: (?:[0-9]+ )?(.*)$" line)
      [(list _ form)
       (match (read (open-input-string form))
         [`(file ,path) (simplify-path path)]
         [_ #f])]
      [_ #f]))

  (test-case "under the package build server, raco test runs only pkg-build/smoke.rkt"
    (define runs
      (filter-map module-run
                  (racket-lines (environment #:service? #t)
                                "-l-" "raco" "test" "-j" "1" "-s" "server-test-set-listing"
                                (path->string (simplify-path collection)))))
    (check-equal? runs (list (simplify-path smoke))))

  (test-case "elsewhere the collection omits nothing from raco test"
    (check-equal?
     (racket-lines (environment #:service? #f)
                   "-l" "racket/base" "-l" "setup/getinfo" "-e"
                   (format "(write ((get-info/full ~s) 'test-omit-paths))"
                           (path->string (simplify-path collection))))
     '("()"))))
