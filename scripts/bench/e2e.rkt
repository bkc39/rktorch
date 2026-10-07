#lang racket/base

(require (only-in compiler/cm managed-compile-zo)
         (only-in compiler/find-exe find-exe)
         (only-in racket/file
                  delete-directory/files make-temporary-directory make-temporary-file)
         (only-in racket/format ~a)
         (only-in racket/match match-define)
         ;; whole-module require on purpose (only-in breaks its expansion)
         racket/runtime-path
         (only-in racket/system process*/ports)
         (only-in "harness.rkt" bench-case)
         (only-in "training.rkt" analyze-steps read-steps))

(provide (struct-out example)
         examples
         find-examples
         example-env
         cache-env
         run-example
         e2e-cases)

(define-runtime-path examples-dir "../../examples/test")
(define-runtime-path recorder "training.rkt")

(struct example (name short full epochs-var warmup) #:transparent)

(define examples
  (list (example "05-mnist" '(("EPOCHS" . 3)) '(("EPOCHS" . 3)) "EPOCHS" 0)
        (example "06-gpt" '(("STEPS" . 300)) '(("STEPS" . 2000)) #f 50)
        (example "08-diffusion" '(("EPOCHS" . 2)) '(("EPOCHS" . 10)) "EPOCHS" 0)
        (example "09-resnet" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0)
        (example "10-dcgan" '(("EPOCHS" . 3)) '(("EPOCHS" . 5)) "EPOCHS" 0)
        (example "11-vae" '(("EPOCHS" . 3)) '(("EPOCHS" . 10)) "EPOCHS" 0)
        (example "12-char-rnn" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0)
        (example "13-translation" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0)
        (example "15-finetune"
                 '(("FEATURE_EPOCHS" . 0) ("FINETUNE_EPOCHS" . 3))
                 '(("FEATURE_EPOCHS" . 0) ("FINETUNE_EPOCHS" . 15))
                 "FINETUNE_EPOCHS" 0)
        (example "16-style-transfer" '(("STEPS" . 100)) '(("STEPS" . 1000)) #f 20)))

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
      (for/list ([v (in-list cache-variables)]
                 #:unless (getenv (car v)))
        (cons (car v) (path->string (build-path root (cdr v)))))
      '()))

(define (child-environment settings)
  (define env (environment-variables-copy (current-environment-variables)))
  (for ([s (in-list settings)])
    (environment-variables-set! env (string->bytes/utf-8 (car s))
                                (string->bytes/utf-8 (~a (cdr s)))))
  env)

(define (run-child runner out settings log)
  (define form
    `(begin
       (require (only-in (file ,(path->string recorder)) record-steps!))
       (record-steps! ,(path->string runner) ,(path->string out))))
  (define dir (make-temporary-directory "rktorch-bench-~a"))
  (define code
    (parameterize ([current-environment-variables (child-environment settings)]
                   [current-directory dir])
      (match-define (list _out _in _pid _err control)
        (process*/ports log (open-input-bytes #"") log
                        (find-exe) "-l" "racket/base" "-e" (format "~s" form)))
      (control 'wait)
      (control 'exit-code)))
  (delete-directory/files dir)
  code)

(define (run-example ex
                     #:runner [runner (build-path examples-dir (string-append (example-name ex) ".rkt"))]
                     #:scale [scale 'short]
                     #:settings [extra '()]
                     #:log [log (current-error-port)])
  (define env
    (append (for/list ([s (in-list (example-env ex scale))]
                       #:unless (assoc (car s) extra))
              s)
            extra))
  (define out (make-temporary-file "rktorch-steps-~a.json"))
  (delete-file out)
  (managed-compile-zo runner)
  (define code (run-child runner out env log))
  (define-values (events wall)
    (if (file-exists? out) (read-steps out) (values '() #f)))
  (when (file-exists? out) (delete-file out))
  (define epochs (let ([v (example-epochs-var ex)]) (and v (cdr (assoc v env)))))
  (define analysis (analyze-steps events #:epochs epochs #:warmup (example-warmup ex)))
  (hash-set* analysis
             'exit_code code
             'wall_s (and wall (/ wall 1000.0))
             'settings (for/hasheq ([s (in-list env)]) (values (string->symbol (car s)) (~a (cdr s))))))

(define (primary-ms ex analysis)
  (cond
    [(hash-ref analysis 'error #f)
     (raise-user-error 'e2e "~a: ~a (exit code ~a)" (example-name ex)
                       (hash-ref analysis 'error) (hash-ref analysis 'exit_code))]
    [(pair? (hash-ref analysis 'epoch_s '()))
     (define epochs (hash-ref analysis 'epoch_s))
     (/ (* 1000.0 (apply + epochs)) (length epochs))]
    [else (hash-ref (hash-ref analysis 'step_ms) 'median)]))

(define (e2e-cases exs
                   #:scale [scale 'short]
                   #:variants [variants '((#f))]
                   #:on-run [on-run void]
                   #:log [log (current-error-port)]
                   #:runner-for [runner-for #f])
  (for*/list ([ex (in-list exs)] [v (in-list variants)])
    (define name (if (car v) (format "~a/~a" (example-name ex) (car v)) (example-name ex)))
    (bench-case (string->symbol name)
                (lambda (_reps)
                  (define analysis
                    (if runner-for
                        (run-example ex #:runner (runner-for ex) #:scale scale
                                     #:settings (cdr v) #:log log)
                        (run-example ex #:scale scale #:settings (cdr v) #:log log)))
                  (on-run ex (car v) analysis)
                  (primary-ms ex analysis))
                #:self-timed? #t)))

(module+ test
  (require (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "harness.rkt" measure result-name result-per-call))

  (define-runtime-path stepper "fixtures/stepper.rkt")
  (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0))
  (define quiet (open-output-nowhere))

  (test-case "examples are found by their number"
    (check-equal? (map example-name (find-examples '("05" "16")))
                  '("05-mnist" "16-style-transfer"))
    (check-exn #rx"no example named 99" (lambda () (find-examples '("99"))))
    (check-equal? (example-env (car examples) 'full) '(("EPOCHS" . 3)))
    (check-equal? (cdr (assoc "STEPS" (example-env (cadr examples) 'short))) 300))

  (test-case "a cache root fills every unset cache variable"
    (check-equal? (cache-env #f) '())
    (parameterize ([current-environment-variables (make-environment-variables
                                                   #"RKTORCH_MNIST_DIR" #"/m")])
      (define env (cache-env "/c"))
      (check-false (assoc "RKTORCH_MNIST_DIR" env))
      (check-equal? (cdr (assoc "RKTORCH_CIFAR10_DIR" env)) "/c/cifar10")))

  (test-case "a child run is timed from its optimizer's steps"
    (define a (run-example fixture #:runner stepper #:scale 'full #:log quiet))
    (check-equal? (hash-ref a 'exit_code) 0)
    (check-equal? (hash-ref a 'steps) 12)
    (check-equal? (hash-ref a 'steps_per_epoch) 3)
    (check-equal? (length (hash-ref a 'epoch_s)) 3)
    (check-equal? (hash-ref (hash-ref a 'settings) 'EPOCHS) "4")
    (check-true (positive? (hash-ref a 'wall_s))))

  (test-case "variants alternate, and a failed run raises"
    (define seen '())
    (define cases
      (e2e-cases (list fixture)
                 #:variants '(("a") ("b" ("EPOCHS" . 2)))
                 #:runner-for (lambda (_ex) stepper)
                 #:log quiet
                 #:on-run (lambda (_ex v a) (set! seen (cons (cons v (hash-ref a 'steps)) seen)))))
    (define results (measure cases #:warmup 0 #:rounds 2 #:load void))
    (check-equal? (map result-name results) '(stepper/a stepper/b))
    (check-equal? (reverse seen) '(("a" . 9) ("b" . 6) ("b" . 6) ("a" . 9)))
    (check-true (andmap positive? (result-per-call (car results))))
    (define broken (e2e-cases (list fixture) #:variants '((#f ("EPOCHS" . 1)))
                              #:runner-for (lambda (_ex) stepper) #:log quiet))
    (check-exn #rx"stepper: no measured steps"
               (lambda () (measure broken #:warmup 0 #:rounds 1 #:load void)))))
