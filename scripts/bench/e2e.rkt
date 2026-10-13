#lang racket/base

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
         (only-in racket/file file->string make-temporary-directory)
         (only-in racket/format ~a)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/runtime-path define-runtime-path)
         (only-in racket/system process*/ports)
         (only-in threading ~>)
         (only-in "harness.rkt" bench-case with-temporary-path)
         (only-in "training.rkt" analyze-steps read-steps))

(provide (struct-out example)
         (struct-out exn:fail:user:e2e)
         examples
         find-examples
         example-env
         control-variables
         merged-settings
         settings->jsexpr
         cache-env
         example-runner
         run-example
         e2e-cases)

(define-runtime-path examples-dir "../../examples/test")
(define-runtime-path recorder "training.rkt")

(struct example (name short full epochs-var warmup optimizer) #:transparent)

(struct exn:fail:user:e2e exn:fail:user (variant exit-code meta))

(define examples
  (list (example "05-mnist" '(("EPOCHS" . 3)) '(("EPOCHS" . 3)) "EPOCHS" 0 'first)
        (example "06-gpt" '(("STEPS" . 300)) '(("STEPS" . 2000)) #f 50 'first)
        (example "08-diffusion" '(("EPOCHS" . 2)) '(("EPOCHS" . 10)) "EPOCHS" 0 'first)
        (example "09-resnet" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0 'first)
        (example "10-dcgan" '(("EPOCHS" . 3)) '(("EPOCHS" . 5)) "EPOCHS" 0 'first)
        (example "11-vae" '(("EPOCHS" . 3)) '(("EPOCHS" . 10)) "EPOCHS" 0 'first)
        (example "12-char-rnn" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0 'first)
        (example "13-translation" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0 'first)
        (example "15-finetune"
                 '(("FEATURE_EPOCHS" . 0) ("FINETUNE_EPOCHS" . 3))
                 '(("FEATURE_EPOCHS" . 0) ("FINETUNE_EPOCHS" . 15))
                 "FINETUNE_EPOCHS" 0 'last)
        (example "16-style-transfer" '(("STEPS" . 100)) '(("STEPS" . 1000)) #f 20 'first)))

(define (find-examples names)
  (for/list ([n (in-list names)])
    (or (for/first ([ex (in-list examples)]
                    #:when (regexp-match? (regexp (string-append "^" (regexp-quote n)))
                                          (example-name ex)))
          ex)
        (raise-user-error 'e2e "no example named ~a" n))))

(define (example-env ex scale)
  (if (eq? scale 'full) (example-full ex) (example-short ex)))

(define cache-variables
  '(("RKTORCH_MNIST_DIR" . "mnist") ("RKTORCH_CIFAR10_DIR" . "cifar10")
    ("RKTORCH_TEXT_DIR" . "text") ("RKTORCH_TRANSLATION_DIR" . "translation")
    ("RKTORCH_HYMENOPTERA_DIR" . "hymenoptera") ("RKTORCH_WEIGHTS_DIR" . "weights")))

(define (cache-env root)
  (if root
      (for/list ([v (in-list cache-variables)])
        (cons (car v) (~> (build-path root (cdr v)) path->complete-path simplify-path
                          path->string)))
      '()))

(define (path-variable? name)
  (or (member name '("PLTUSERHOME" "PLTADDONDIR"))
      (regexp-match? #rx"^RKTORCH_.*_DIR$" name)))

(define (resolved setting)
  (match setting
    [(cons name (? path-string? value))
     #:when (path-variable? name)
     (cons name (~> value path->complete-path simplify-path path->string))]
    [_ setting]))

(define (merged-settings ex scale extra)
  (map resolved
       (append (for/list ([s (in-list (example-env ex scale))]
                          #:unless (assoc (car s) extra))
                 s)
               extra)))

(define (settings->jsexpr settings)
  (for/hasheq ([s (in-list settings)]) (values (string->symbol (car s)) (~a (cdr s)))))

(define (control-variables ex runner)
  (remove-duplicates
   (append (map car (example-short ex))
           (map car (example-full ex))
           (regexp-match* #px"\\((?:env-number|getenv) \"([^\"]+)\"" (file->string runner)
                          #:match-select cadr))))

(define (child-environment cleared settings)
  (define env (environment-variables-copy (current-environment-variables)))
  (for ([name (in-list cleared)])
    (environment-variables-set! env (string->bytes/utf-8 name) #f))
  (for ([s (in-list settings)])
    (environment-variables-set! env (string->bytes/utf-8 (car s))
                                (~> s cdr ~a string->bytes/utf-8)))
  env)

(define (run-child runner dir out cleared settings log)
  (define form
    `(begin
       (require (only-in (file ,(path->string recorder)) record-steps!))
       (record-steps! ,(path->string runner) ,(path->string out))))
  (parameterize ([current-environment-variables (child-environment cleared settings)]
                 [current-directory dir])
    (match-define (list _out _in _pid _err control)
      (process*/ports log (open-input-bytes #"") log
                      (find-exe) "-l" "racket/base" "-e" (format "~s" form)))
    (control 'wait)
    (control 'exit-code)))

(define (example-runner ex)
  (build-path examples-dir (string-append (example-name ex) ".rkt")))

(define (run-example ex
                     #:runner [runner (example-runner ex)]
                     #:scale [scale 'short]
                     #:settings [extra '()]
                     #:log [log (current-error-port)])
  (define env (merged-settings ex scale extra))
  (define-values (code events wall meta)
    (with-temporary-path (dir (make-temporary-directory "rktorch-bench-~a"))
      (define out (build-path dir "rktorch-steps.json"))
      (define code (run-child runner dir out (control-variables ex runner) env log))
      (define-values (events wall meta)
        (if (file-exists? out) (read-steps out) (values '() #f #f)))
      (values code events wall meta)))
  (define epoch-setting (and (example-epochs-var ex) (assoc (example-epochs-var ex) env)))
  (define epochs (and epoch-setting (~> epoch-setting cdr ~a string->number)))
  (define analysis (analyze-steps events #:epochs epochs #:warmup (example-warmup ex)
                                  #:optimizer (example-optimizer ex)))
  (hash-set* analysis
             'exit_code code
             'wall_s (and wall (/ wall 1000.0))
             'meta meta
             'settings (settings->jsexpr env)))

(define (primary-ms ex variant analysis)
  (cond
    [(hash-ref analysis 'error #f)
     (raise (exn:fail:user:e2e
             (format "e2e: ~a: ~a (exit code ~a)" (example-name ex)
                     (hash-ref analysis 'error) (hash-ref analysis 'exit_code))
             (current-continuation-marks)
             variant
             (hash-ref analysis 'exit_code)
             (hash-ref analysis 'meta #f)))]
    [(pair? (hash-ref analysis 'epoch_s '()))
     (define epochs (hash-ref analysis 'epoch_s))
     (/ (* 1000.0 (apply + epochs)) (length epochs))]
    [else (/ 1000.0 (hash-ref analysis 'steps_per_s))]))

(define (e2e-cases exs
                   #:scale [scale 'short]
                   #:variants [variants '((#f))]
                   #:on-run [on-run void]
                   #:log [log (current-error-port)]
                   #:runner-for [runner-for example-runner])
  (for*/list ([ex (in-list exs)] [v (in-list variants)])
    (define name (if (car v) (format "~a/~a" (example-name ex) (car v)) (example-name ex)))
    (bench-case (string->symbol name)
                (lambda (_reps)
                  (define analysis
                    (run-example ex #:runner (runner-for ex) #:scale scale
                                 #:settings (cdr v) #:log log))
                  (on-run ex (car v) analysis)
                  (primary-ms ex (car v) analysis))
                #:self-timed? #t)))

(module+ test
  (require (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "harness.rkt" measure result-name result-per-call))

  (define-runtime-path stepper "fixtures/stepper.rkt")
  (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0 'first))
  (define quiet (open-output-nowhere))

  (test-case "examples are found by their number"
    (check-equal? (map example-name (find-examples '("05" "16")))
                  '("05-mnist" "16-style-transfer"))
    (check-exn #rx"no example named 99" (lambda () (find-examples '("99"))))
    (check-equal? (example-env (car examples) 'full) '(("EPOCHS" . 3)))
    (check-equal? (cdr (assoc "STEPS" (example-env (cadr examples) 'short))) 300))

  (test-case "a cache root sets every cache variable, inherited or not"
    (check-equal? (cache-env #f) '())
    (parameterize ([current-environment-variables (make-environment-variables
                                                   #"RKTORCH_MNIST_DIR" #"/m")])
      (define env (cache-env "/c"))
      (check-equal? (length env) 6)
      (check-equal? (cdr (assoc "RKTORCH_MNIST_DIR" env)) "/c/mnist")
      (check-equal? (cdr (assoc "RKTORCH_CIFAR10_DIR" env)) "/c/cifar10")
      (check-true (absolute-path? (cdr (assoc "RKTORCH_TEXT_DIR" (cache-env "rel")))))))

  (test-case "a variant's relative paths are the caller's"
    (define settings
      (merged-settings fixture 'short '(("PLTUSERHOME" . "other/.racket-user")
                                        ("RKTORCH_MNIST_DIR" . "data")
                                        ("OMP_NUM_THREADS" . "1"))))
    (check-equal? (cdr (assoc "PLTUSERHOME" settings))
                  (path->string (build-path (current-directory) "other" ".racket-user")))
    (check-equal? (cdr (assoc "RKTORCH_MNIST_DIR" settings))
                  (path->string (build-path (current-directory) "data")))
    (check-equal? (cdr (assoc "OMP_NUM_THREADS" settings)) "1")
    (check-equal? (cdr (assoc "EPOCHS" settings)) 3))

  (test-case "a child run is timed from its optimizer's steps"
    (define a (run-example fixture #:runner stepper #:scale 'full #:log quiet))
    (check-equal? (hash-ref a 'exit_code) 0)
    (check-equal? (hash-ref a 'steps) 12)
    (check-equal? (hash-ref a 'steps_per_epoch) 3)
    (check-equal? (length (hash-ref a 'epoch_s)) 3)
    (check-equal? (hash-ref (hash-ref a 'settings) 'EPOCHS) "4")
    (check-true (positive? (hash-ref a 'wall_s)))
    (check-true (string? (hash-ref (hash-ref a 'meta) 'libtorch))))

  (test-case "a child sees only the example settings it is given"
    (check-equal? (control-variables (cadr examples) (example-runner (cadr examples)))
                  '("STEPS" "EXCERPT"))
    (check-equal? (control-variables fixture stepper) '("EPOCHS" "STEPS"))
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (putenv "STEPS" "5")
      (putenv "EPOCHS" "1")
      (check-equal? (hash-ref (run-example fixture #:runner stepper #:log quiet) 'steps) 9)
      (define given (run-example fixture #:runner stepper #:settings '(("STEPS" . "7"))
                                 #:log quiet))
      (check-equal? (hash-ref given 'steps) 7)
      (check-equal? (hash-ref (hash-ref given 'settings) 'STEPS) "7")))

  (test-case "variants alternate, and a failed run raises"
    (define seen '())
    (define cases
      (e2e-cases (list fixture)
                 #:variants '(("a") ("b" ("EPOCHS" . "2")))
                 #:runner-for (lambda (_ex) stepper)
                 #:log quiet
                 #:on-run (lambda (_ex v a) (set! seen (cons (cons v (hash-ref a 'steps)) seen)))))
    (define results (measure cases #:warmup 0 #:rounds 2 #:load void))
    (check-equal? (map result-name results) '(stepper/a stepper/b))
    (check-equal? (reverse seen) '(("a" . 9) ("b" . 6) ("b" . 6) ("a" . 9)))
    (check-true (andmap positive? (result-per-call (car results))))
    (define broken (e2e-cases (list fixture) #:variants '((#f ("EPOCHS" . 1)))
                              #:runner-for (lambda (_ex) stepper) #:log quiet))
    (check-exn (lambda (e)
                 (and (exn:fail:user:e2e? e)
                      (regexp-match? #rx"stepper: no measured steps" (exn-message e))
                      (not (exn:fail:user:e2e-variant e))
                      (equal? (exn:fail:user:e2e-exit-code e) 0)))
               (lambda () (measure broken #:warmup 0 #:rounds 1 #:load void)))))
