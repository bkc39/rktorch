#lang racket/base

(module+ test
  (require racket/list
           racket/match
           racket/path
           racket/set
           rackunit
           threading)

  (define ns (make-base-namespace))

  (define (expands? form)
    (parameterize ([current-namespace ns])
      (expand form)
      #t))

  (define (export-names mod phase-ok?)
    (parameterize ([current-namespace ns])
      (module-declared? mod #t)
      (define-values (variables syntaxes) (module->exports mod))
      (for*/seteq ([phase+exports (in-list (append variables syntaxes))]
                   #:when (phase-ok? (car phase+exports))
                   [export (in-list (cdr phase+exports))])
        (car export))))

  (define (any-phase _) #t)
  (define (run-time phase) (eqv? phase 0))

  (define torch-dir
    (path-only (collection-file-path "main.rkt" "torch")))

  (define (descend? dir)
    (not (member (path->string (file-name-from-path dir))
                 '("compiled" "fixtures"))))

  (define (public-module relative)
    (match (map path->string (explode-path relative))
      [(list "info.rkt") #f]
      [(or (list (regexp #rx"[.]rkt$"))
           (cons (or "audio" "data" "text" "vision")
                 (app last (regexp #rx"[.]rkt$"))))
       (~>> (path-replace-extension relative #"")
            path->string
            (string-append "torch/")
            string->symbol)]
      [_ #f]))

  (define public-modules
    (parameterize ([current-directory torch-dir])
      (sort (filter-map public-module
                        (for/list ([relative (in-directory #f descend?)]
                                   #:when (file-exists? relative))
                          relative))
            symbol<?)))

  (test-case "L7: the label helper exports racket, torch and torch/nn"
    (check-equal? (export-names 'torch/scribblings/labels not)
                  (set-union (export-names 'racket any-phase)
                             (export-names 'torch run-time)
                             (export-names 'torch/nn run-time))))

  (test-case "L7: the label helper exports nothing outside the label phase"
    (check-true (set-empty? (export-names 'torch/scribblings/labels values))))

  (test-case "L7: the walk finds the public modules"
    (check-not-false (memq 'torch/main public-modules))
    (check-not-false (memq 'torch/data/loader public-modules))
    (check-not-false (memq 'torch/vision/mnist public-modules))
    (check-false (memq 'torch/info public-modules)))

  (define (composes? . requires)
    (with-handlers ([exn:fail:syntax? (lambda (_) #f)])
      (expands? `(module composed '#%kernel (#%require ,@requires)))))

  (test-case "L7: the helper composes with racket's libraries and threading"
    (check-true
     (composes? 'torch/scribblings/labels
                '(for-label torch torch/nn threading
                            racket/contract racket/contract/base
                            racket/match racket/sequence racket/string
                            torch/data/loader torch/text/translation
                            torch/vision/mnist))))

  (test-case "L7: the helper composes with each public module torch does"
    (for ([module (in-list public-modules)])
      (check-equal? (composes? 'torch/scribblings/labels `(for-label ,module))
                    (composes? `(for-label torch torch/nn ,module))
                    (symbol->string module))))

  (test-case "L7: the composition check rejects racket/base beside the helper"
    (check-exn #rx"already required for label"
               (lambda ()
                 (expands?
                  '(module clashing '#%kernel
                     (#%require torch/scribblings/labels
                                (for-label racket/base))))))))
