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
                 (app last (regexp #rx"[.]rkt$")))
           (list* "foreign" "raw" (app last (regexp #rx"[.]rkt$"))))
       (~>> (path-replace-extension relative #"")
            path->string
            (string-append "torch/")
            string->symbol)]
      [_ #f]))

  (define public-modules
    (parameterize ([current-directory torch-dir])
      (~> (for/list ([relative (in-directory #f descend?)]
                     #:when (file-exists? relative))
            relative)
          (filter-map public-module _)
          (sort symbol<?))))

  (test-case "the label helper exports racket, torch and torch/nn"
    (check-equal? (export-names 'torch/scribblings/labels not)
                  (set-union (export-names 'racket any-phase)
                             (export-names 'torch run-time)
                             (export-names 'torch/nn run-time))))

  (test-case "the label helper exports nothing outside the label phase"
    (check-true (set-empty? (export-names 'torch/scribblings/labels values))))

  (test-case "the walk finds the public modules and skips info.rkt"
    (check-true (subset? (seteq 'torch/main 'torch/data/loader 'torch/vision/mnist
                                'torch/foreign/raw/tensor)
                         (list->seteq public-modules)))
    (check-false (member 'torch/info public-modules)))

  (define (composes? . requires)
    (with-handlers ([exn:fail:syntax? (lambda (_) #f)])
      (expands? `(module composed '#%kernel (#%require ,@requires)))))

  (test-case "the helper composes with racket's libraries and threading"
    (check-true
     (composes? 'torch/scribblings/labels
                '(for-label torch torch/nn threading
                            racket/contract racket/contract/base
                            racket/match racket/sequence racket/string
                            torch/data/loader torch/text/translation
                            torch/vision/mnist))))

  (define clashing-modules
    (for/list ([module (in-list public-modules)]
               #:unless (composes? `(for-label torch torch/nn ,module)))
      module))

  (test-case "only torch/generated and three raw FFI modules clash for label"
    (check-equal? clashing-modules
                  '(torch/foreign/raw/memory
                    torch/foreign/raw/pressure
                    torch/foreign/raw/pressure-settings
                    torch/generated)))

  (test-case "the helper composes with every public module but the clashing ones"
    (for ([module (in-list public-modules)]
          #:unless (memq module clashing-modules))
      (check-true (composes? 'torch/scribblings/labels `(for-label ,module))
                  (symbol->string module))))

  (define (label-namespace spec)
    (define label-ns (make-base-empty-namespace))
    (parameterize ([current-namespace label-ns])
      (namespace-require spec))
    label-ns)

  (define (label-source label-ns name)
    (parameterize ([current-namespace label-ns])
      (match (identifier-label-binding (namespace-symbol->identifier name))
        [(list* source-module source-name _)
         (list (module-path-index-resolve source-module) source-name)]
        [_ #f])))

  (test-case "each name torch shares with racket labels torch's binding"
    (define helper-ns (label-namespace 'torch/scribblings/labels))
    (define torch-ns (label-namespace '(for-label torch torch/nn)))
    (define racket-ns (label-namespace '(for-label racket)))
    (define shared
      (set-intersect (set-union (export-names 'torch run-time)
                                (export-names 'torch/nn run-time))
                     (export-names 'racket any-phase)))
    (check-false (set-empty? shared))
    (for ([name (in-set shared)])
      (define source (label-source helper-ns name))
      (check-not-false source (symbol->string name))
      (check-equal? source (label-source torch-ns name) (symbol->string name))
      (check-not-equal? source (label-source racket-ns name)
                        (symbol->string name))))

  (test-case "racket/base beside the helper is refused as already labelled"
    (check-exn #rx"already required for label"
               (lambda ()
                 (expands?
                  '(module clashing '#%kernel
                     (#%require torch/scribblings/labels
                                (for-label racket/base))))))))
