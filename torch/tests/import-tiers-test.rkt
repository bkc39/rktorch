#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file file->lines)
           (only-in racket/list filter-map last)
           (only-in racket/match match)
           (only-in racket/path file-name-from-path path-only)
           (only-in racket/runtime-path define-runtime-path)
           (only-in racket/set
                    in-set list->seteq set set->list set-add set-empty?
                    set-intersect set-member? set-union seteq subset?)
           (only-in rackunit
                    check-equal? check-exn check-false check-not-eq?
                    check-not-equal? check-not-false check-true test-case)
           (only-in threading ~> ~>>)
           (only-in torch/private/exports export-names))

  (define ns (make-base-namespace))

  (define (expands? form)
    (parameterize ([current-namespace ns])
      (expand form)
      #t))

  (define (export-set module . phase-ok?)
    (parameterize ([current-namespace ns])
      (list->seteq (apply export-names module phase-ok?))))

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
    (check-equal? (export-set 'torch/scribblings/labels not)
                  (set-union (export-set 'racket)
                             (export-set 'torch run-time)
                             (export-set 'torch/nn run-time))))

  (test-case "the label helper exports nothing outside the label phase"
    (check-true (set-empty? (export-set 'torch/scribblings/labels values))))

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
      (set-intersect (set-union (export-set 'torch run-time)
                                (export-set 'torch/nn run-time))
                     (export-set 'racket)))
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
                                (for-label racket/base)))))))

  (define torch-exports (export-set 'torch))
  (define safe-exports (export-set 'torch/safe))
  (define operator-exports (export-set 'torch/operators))
  (define racket-exports (export-set 'racket))

  (test-case "L1: torch/safe and torch/operators partition torch"
    (check-equal? (set-union safe-exports operator-exports) torch-exports)
    (check-true (set-empty? (set-intersect safe-exports operator-exports)))
    (check-true (composes? 'torch 'torch/safe 'torch/operators)))

  (test-case "L2: torch/operators is what torch shares with racket"
    (check-false (set-empty? operator-exports))
    (check-equal? operator-exports (set-intersect torch-exports racket-exports)))

  (test-case "L3: torch/safe requires beside racket and threading"
    (check-true (composes? 'racket 'threading 'torch/safe))
    (check-true (composes? '(for-label racket threading torch/safe))))

  (test-case "L3': torch beside racket/list or racket/base is refused"
    (check-exn #rx"identifier already required"
               (lambda ()
                 (expands? '(module clashing '#%kernel
                              (#%require racket/list torch)))))
    (check-exn #rx"already required for label"
               (lambda ()
                 (expands? '(module clashing '#%kernel
                              (#%require (for-label racket/base torch)))))))

  (define shadowing-modules '(torch/foreign torch/main torch/operators))

  (test-case "L4: every other public module requires beside racket"
    (check-true (subset? (seteq 'torch/generated 'torch/nn 'torch/safe)
                         (list->seteq public-modules)))
    (for ([module (in-list public-modules)]
          #:unless (memq module shadowing-modules))
      (check-true (composes? 'racket 'threading module)
                  (symbol->string module))
      (check-true (composes? `(for-label racket threading ,module))
                  (symbol->string module))))

  (test-case "L5: the public modules require together"
    (check-true
     (apply composes? '(submod torch/foreign unsafe)
            (remq* clashing-modules public-modules))))

  (define fallback-rows
    '((+ () (1) (1 2 3) (1/2 0.5))
      (- (1) (5 2 1))
      (* () (2 3 4))
      (/ (2) (1 2 4))
      (< (1 2) (1 2 3) (2 1))
      (<= (1 1) (1 2 2))
      (> (2 1) (3 2 1))
      (>= (1 1) (2 2 1))
      (abs (-3) (-1/2))
      (argmax (car '((1 a) (3 b))))
      (cos (0) (1.0))
      (exp (0) (1.0))
      (flatten ('(1 (2 (3)))))
      (length ('(1 2 3)))
      (log (1) (8 2) (2.0))
      (max (1 2.0 3))
      (min (1 2 3))
      (sin (0) (1.0))
      (sort ('(3 1 2) <))
      (sqrt (4) (-4) (2.0))
      (take ('(1 2 3) 2))
      (tanh (0) (0.5))))

  (define (argument-values arguments)
    (parameterize ([current-namespace ns])
      (eval `(list ,@arguments))))

  (define (operator-pair name)
    (parameterize ([current-namespace ns])
      (values (dynamic-require 'torch/operators name)
              (dynamic-require 'racket name))))

  (test-case "L6: the fallback table has a row for each operator"
    (check-equal? (list->seteq (map car fallback-rows)) operator-exports))

  (test-case "L6: each operator falls back to racket's on plain values"
    (for* ([row (in-list fallback-rows)]
           #:when (set-member? operator-exports (car row))
           [arguments (in-list (cdr row))])
      (define-values (torch-operator racket-operator) (operator-pair (car row)))
      (define plain (argument-values arguments))
      (define message (format "~s" (cons (car row) arguments)))
      (check-not-eq? torch-operator racket-operator message)
      (check-equal? (apply torch-operator plain)
                    (apply racket-operator plain)
                    message)))

  (define (rebase import self)
    (define-values (path base) (module-path-index-split import))
    (cond
      [(not path) self]
      [(module-path-index? base) (module-path-index-join path (rebase base self))]
      [else (module-path-index-join path base)]))

  (define (static-imports root)
    (let walk ([pending (list (module-path-index-resolve
                               (module-path-index-join root #f) #t))]
               [seen (set)])
      (match pending
        ['() seen]
        [(cons name rest)
         #:when (set-member? seen name)
         (walk rest seen)]
        [(cons name rest)
         (walk (append (for*/list ([phase+imports (in-list (module->imports name))]
                                   #:when (car phase+imports)
                                   [import (in-list (cdr phase+imports))])
                         (module-path-index-resolve (rebase import name) #t))
                       rest)
               (set-add seen name))])))

  (define racket-main
    (make-resolved-module-path (collection-file-path "main.rkt" "racket")))

  (test-case "L8: torch/safe's static imports do not reach racket"
    (parameterize ([current-namespace (make-base-namespace)])
      (eval '(module requires-racket racket/base (require racket)))
      (check-true (set-member? (static-imports ''requires-racket) racket-main))
      (check-false (set-member? (static-imports 'torch/safe) racket-main))))

  (define-runtime-path racket-names-file "../../codegen/racket-names.txt")

  (test-case "codegen/racket-names.txt is racket's and threading's exports"
    (check-equal? (file->lines racket-names-file)
                  (~> (set-union racket-exports (export-set 'threading))
                      set->list
                      (sort symbol<?)
                      (map symbol->string _)))))
