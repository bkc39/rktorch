#lang racket/base

(module+ test
  (require (only-in racket/file make-directory* make-temporary-directory)
           (only-in racket/match match-define)
           (only-in racket/port with-output-to-string)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-replace string-trim)
           (only-in racket/system system*)
           rackunit
           (only-in "harness.rkt" with-temporary-path))

  (define-runtime-path writer "cifar-folder.rkt")
  (define-runtime-path resolver "python/cache_dirs.py")

  (define host-platform (if (eq? (system-type 'os) 'macosx) "darwin" "linux"))

  ;; (platform environment directories-made expected), "@" the user's home
  (define cases
    '(("linux" () () "@/.cache/racket/rktorch/cifar10-png")
      ("linux" () (".racket") "@/.racket/rktorch/cifar10-png")
      ("linux" () (".racket" ".cache/racket") "@/.cache/racket/rktorch/cifar10-png")
      ("linux" (("XDG_CACHE_HOME" . "@/x")) () "@/x/racket/rktorch/cifar10-png")
      ("linux" (("XDG_CACHE_HOME" . "@/x")) (".racket") "@/.racket/rktorch/cifar10-png")
      ("linux" (("XDG_CACHE_HOME" . "rel")) () "@/.cache/racket/rktorch/cifar10-png")
      ("linux" (("PLTUSERHOME" . "@/pu") ("XDG_CACHE_HOME" . "@/x")) ("pu" "x/racket")
               "@/pu/.cache/racket/rktorch/cifar10-png")
      ("linux" (("PLTUSERHOME" . "@/pu")) ("pu/.racket") "@/pu/.racket/rktorch/cifar10-png")
      ("linux" (("RKTORCH_CIFAR10_PNG_DIR" . "@/png")) () "@/png")
      ("darwin" (("XDG_CACHE_HOME" . "@/x")) (".racket" ".cache/racket" "x/racket")
                "@/Library/Caches/Racket/rktorch/cifar10-png")
      ("darwin" (("PLTUSERHOME" . "@/pu")) ("pu") "@/pu/Library/Caches/Racket/rktorch/cifar10-png")
      ("darwin" (("RKTORCH_CIFAR10_PNG_DIR" . "@/png")) () "@/png")))

  (define cleared '(#"HOME" #"PLTUSERHOME" #"XDG_CACHE_HOME" #"RKTORCH_CIFAR10_PNG_DIR"))

  (define (environment home settings)
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([name (in-list cleared)])
      (environment-variables-set! env name #f))
    (environment-variables-set! env #"PLTADDONDIR"
                                (path->bytes (find-system-path 'addon-dir)))
    (for ([s (in-list (cons (cons "HOME" "@") settings))])
      (environment-variables-set! env (string->bytes/utf-8 (car s))
                                  (string->bytes/utf-8 (string-replace (cdr s) "@" home))))
    env)

  (define (output env program . args)
    (parameterize ([current-environment-variables env])
      (string-trim (with-output-to-string (lambda () (apply system* program args))))))

  (define (racket-writes env)
    (output env (find-executable-path (find-system-path 'exec-file)) "-l" "racket/base" "-e"
            (format "~s" `(begin (require (only-in (file ,(path->string writer)) cifar10-folder))
                                 (display (cifar10-folder "full"))))))

  (define (python-reads python env platform)
    (output env python "-P" (path->string resolver) "RKTORCH_CIFAR10_PNG_DIR" "cifar10-png"
            platform))

  (define python (find-executable-path "python3"))

  (cond
    [(not python) (displayln "[cache-dirs-test] skipped: no python3")]
    [else
     (for ([c (in-list cases)])
       (match-define (list platform settings dirs expected) c)
       (test-case (format "the twins read the PNG tree where cifar-folder.rkt writes it: ~s" c)
         (with-temporary-path (root (make-temporary-directory "cache-dirs-~a"))
           (define home (path->string (build-path root "home")))
           (for ([d (in-list (cons "." dirs))])
             (make-directory* (build-path home d)))
           (define env (environment home settings))
           (define want (string-replace expected "@" home))
           (check-equal? (python-reads python env platform) want "Python")
           (when (equal? platform host-platform)
             (check-equal? (racket-writes env) (path->string (build-path want "full"))
                           "Racket")))))]))
