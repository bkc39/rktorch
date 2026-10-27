#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/file file->string make-temporary-directory)
         (only-in racket/format ~a)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/path path-only)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-join)
         (only-in threading ~>)
         (only-in "child.rkt" python-runner? run-child)
         (only-in "e2e-examples.rkt"
                  example-env example-epochs-var example-full example-name example-optimizer
                  example-runner example-short example-warmup)
         (only-in "harness.rkt" bench-case with-temporary-path)
         (only-in "training.rkt" analyze-steps read-steps recorded-run))

(provide (struct-out exn:fail:user:e2e)
         control-variables
         runner-reads
         merged-settings
         settings->jsexpr
         cache-env
         run-example
         e2e-cases)

(struct exn:fail:user:e2e exn:fail:user (example variant exit-code meta))

(define cache-variables
  '(("RKTORCH_MNIST_DIR" . "mnist") ("RKTORCH_CIFAR10_DIR" . "cifar10")
    ("RKTORCH_TEXT_DIR" . "text") ("RKTORCH_TRANSLATION_DIR" . "translation")
    ("RKTORCH_HYMENOPTERA_DIR" . "hymenoptera") ("RKTORCH_WEIGHTS_DIR" . "weights")
    ("RKTORCH_CIFAR10_PNG_DIR" . "cifar10-png")))

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

(define recorder-variables
  '("RKTORCH_BENCH_COLLECT_AT_TROUGHS" "RKTORCH_BENCH_COLLECT_BUDGET"))

(define (runner-sources runner)
  (define text (file->string runner))
  (define siblings
    (for*/list ([name (in-list (regexp-match* #px"\\(only-in \"([a-z0-9-]+[.]rkt)\"" text
                                              #:match-select cadr))]
                [path (in-value (~> runner path->complete-path path-only (build-path name)))]
                #:when (and (not (python-runner? runner)) (file-exists? path)))
      (file->string path)))
  (cons text siblings))

(define (control-variables ex runner)
  (remove-duplicates
   (append (map car (example-short ex))
           (map car (example-full ex))
           recorder-variables
           (for*/list ([text (in-list (runner-sources runner))]
                       [name (in-list (regexp-match*
                                       #px"(?:\\((?:env-number|getenv) |env_(?:number|text)\\()\"([^\"]+)\""
                                       text
                                       #:match-select cadr))])
             name))))

(define (runner-reads ex runner)
  (for/list ([name (in-list (control-variables ex runner))]
             #:unless (and (python-runner? runner) (member name recorder-variables)))
    name))

(define time-limits (hasheq 'short 3600 'full 43200))

(define (step-hook? torch-dir)
  (or (file-exists? (build-path torch-dir "private" "step-log.rkt"))
      (file-exists? (build-path torch-dir "private" "compiled" "step-log_rkt.zo"))))

(define (classify events wall torch-dir
                  #:epochs [epochs #f] #:warmup [warmup 0] #:optimizer [optimizer 'first])
  (define pre-hook? (and torch-dir (not (step-hook? torch-dir))))
  (cond
    [(and (null? events) wall pre-hook?) (hasheq 'steps 0 'timing "wall")]
    [(and (null? events) wall torch-dir)
     (hasheq 'steps 0
             'timing "steps"
             'error (format "torch at ~a has the step hook but logged no steps" torch-dir))]
    [(and (null? events) wall)
     (hasheq 'steps 0 'timing "steps" 'error "logged no steps and named no torch")]
    [else
     (hash-set (analyze-steps events #:epochs epochs #:warmup warmup #:optimizer optimizer)
               'timing "steps")]))

(define no-run (recorded-run '() #f #f #f))

(define (recorded-steps out)
  (with-handlers ([exn:fail? (lambda (e)
                               (values no-run (format "unreadable steps file: ~a" (exn-message e))))])
    (values (if (file-exists? out) (read-steps out) no-run) #f)))

(define (run-example ex
                     #:runner [runner (example-runner ex)]
                     #:scale [scale 'short]
                     #:settings [extra '()]
                     #:time-limit [time-limit #f]
                     #:log [log (current-error-port)])
  (define env (merged-settings ex scale extra))
  (define limit (or time-limit (hash-ref time-limits scale)))
  (define-values (status recorded unreadable)
    (with-temporary-path (dir (make-temporary-directory "rktorch-bench-~a"))
      (define out (build-path dir "rktorch-steps.json"))
      (define status (run-child runner dir out (control-variables ex runner) env log limit))
      (define-values (recorded unreadable) (recorded-steps out))
      (values status recorded unreadable)))
  (match-define (list code killed?) status)
  (match-define (recorded-run events wall meta torch-dir) recorded)
  (define epoch-setting (and (example-epochs-var ex) (assoc (example-epochs-var ex) env)))
  (define epochs (and epoch-setting (~> epoch-setting cdr ~a string->number)))
  (define analysis
    (classify events wall torch-dir
              #:epochs epochs #:warmup (example-warmup ex) #:optimizer (example-optimizer ex)))
  (define exit-failure
    (cond
      [killed? (format "killed after no exit within ~a s" limit)]
      [(zero? code) #f]
      [else "exited non-zero"]))
  (define errors (filter values (list exit-failure unreadable (hash-ref analysis 'error #f))))
  (hash-set* (if (pair? errors) (hash-set analysis 'error (string-join errors "; ")) analysis)
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
             (example-name ex)
             variant
             (hash-ref analysis 'exit_code)
             (hash-ref analysis 'meta #f)))]
    [(equal? (hash-ref analysis 'timing) "wall") (* 1000.0 (hash-ref analysis 'wall_s))]
    [(pair? (hash-ref analysis 'epoch_s '()))
     (define epochs (hash-ref analysis 'epoch_s))
     (/ (* 1000.0 (apply + epochs)) (length epochs))]
    [else (/ 1000.0 (hash-ref analysis 'steps_per_s))]))

(define (e2e-cases exs
                   #:scale [scale 'short]
                   #:variants [variants '((#f))]
                   #:on-run [on-run void]
                   #:time-limit [time-limit #f]
                   #:log [log (current-error-port)]
                   #:runner-for [runner-for example-runner])
  (for*/list ([ex (in-list exs)] [v (in-list variants)])
    (define name (if (car v) (format "~a/~a" (example-name ex) (car v)) (example-name ex)))
    (bench-case (string->symbol name)
                (lambda (_reps)
                  (define analysis
                    (run-example ex #:runner (runner-for ex) #:scale scale
                                 #:settings (cdr v) #:time-limit time-limit #:log log))
                  (on-run ex (car v) analysis)
                  (primary-ms ex (car v) analysis))
                #:self-timed? #t)))

(module+ test
  (require (only-in racket/file make-directory*)
           (only-in racket/port open-output-nowhere)
           (only-in racket/system system*)
           rackunit
           (only-in "e2e-examples.rkt" example examples find-examples)
           (only-in "fixtures/pre-hook-torch.rkt" call-with-pre-hook-torch)
           (only-in "harness.rkt" measure result-name result-per-call)
           (only-in "shim.rkt" torch-dir))

  (define-runtime-path broken "fixtures/broken.rkt")
  (define-runtime-path collect-reader "fixtures/collect-reader.rkt")
  (define-runtime-path grandchild "fixtures/grandchild.rkt")
  (define-runtime-path lingering "fixtures/lingering.rkt")
  (define-runtime-path partial-steps "fixtures/partial-steps.rkt")
  (define-runtime-path sleeper "fixtures/sleeper.rkt")
  (define-runtime-path stepper "fixtures/stepper.rkt")
  (define-runtime-path teardown-crash "fixtures/teardown-crash.rkt")
  (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0 'first))
  (define quiet (open-output-nowhere))

  (define (signal! flag pid)
    (parameterize ([current-output-port quiet] [current-error-port quiet])
      (system* (find-executable-path "kill") flag (~a pid))))

  (define (dead? pid)
    (not (signal! "-0" pid)))

  (define (within? seconds done?)
    (define deadline (+ (current-inexact-monotonic-milliseconds) (* 1000.0 seconds)))
    (let poll ()
      (cond
        [(done?) #t]
        [(> (current-inexact-monotonic-milliseconds) deadline) #f]
        [else (sleep 0.1) (poll)])))

  (define (call-with-pid-file proc)
    (with-temporary-path (dir (make-temporary-directory "bench-pids-~a"))
      (define pid-file (build-path dir "pids"))
      (define (started)
        (if (file-exists? pid-file) (call-with-input-file pid-file read) '()))
      (dynamic-wind
       void
       (lambda () (proc (cons "BENCH_PID_FILE" (path->string pid-file)) pid-file))
       (lambda ()
         (for ([pid (in-list (started))] #:unless (dead? pid))
           (signal! "-KILL" pid))))))

  (define (grandchild-pid pid-file)
    (match-define (list _child pid) (call-with-input-file pid-file read))
    pid)

  (test-case "a PyTorch twin's settings are the ones its script reads"
    (define twin (car (find-examples '("py-17"))))
    (check-equal? (control-variables twin (example-runner twin))
                  (append '("EPOCHS") recorder-variables
                          '("WORKERS" "BATCH" "VARIANT" "DATASET" "OPTIMIZER")))
    (check-equal? (runner-reads twin (example-runner twin))
                  '("EPOCHS" "WORKERS" "BATCH" "VARIANT" "DATASET" "OPTIMIZER"))
    (define per-param (car (find-examples '("py-17-workers-per"))))
    (check-equal? (example-name per-param) "py-17-workers-per-param")
    (check-equal? (cdr (assoc "OPTIMIZER" (example-env per-param 'short))) "per-param")
    (check-equal? (example-runner per-param) (example-runner twin)))

  (test-case "a workload's settings include those its sibling modules read for it"
    (define loader (car (find-examples '("cifar-loader"))))
    (check-equal? (control-variables loader (example-runner loader))
                  (append '("EPOCHS" "BATCH") recorder-variables
                          '("WORKERS" "PIPELINE" "DATASET" "PNG_DIGEST")))
    (define twin (car (find-examples '("py-cifar-loader"))))
    (check-equal? (runner-reads twin (example-runner twin))
                  '("EPOCHS" "BATCH" "WORKERS" "PIPELINE" "DATASET" "PNG_DIGEST")))

  (test-case "a cache root sets every cache variable, inherited or not"
    (check-equal? (cache-env #f) '())
    (parameterize ([current-environment-variables (make-environment-variables
                                                   #"RKTORCH_MNIST_DIR" #"/m")])
      (define env (cache-env "/c"))
      (check-equal? (length env) 7)
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

  (test-case "a child whose torch has the step hook but logs no steps fails"
    (define a (run-example fixture #:runner stepper #:settings '(("STEPPER_SILENT" . "1"))
                           #:log quiet))
    (check-equal? (hash-ref a 'exit_code) 0)
    (check-equal? (hash-ref a 'steps) 0)
    (check-regexp-match #rx"step hook" (hash-ref a 'error #f))
    (define cases
      (e2e-cases (list fixture) #:variants '(("new" ("STEPPER_SILENT" . "1")))
                 #:runner-for (lambda (_ex) stepper) #:log quiet))
    (check-exn (lambda (e)
                 (and (exn:fail:user:e2e? e)
                      (regexp-match? #rx"step hook" (exn-message e))
                      (equal? (exn:fail:user:e2e-variant e) "new")))
               (lambda () (measure cases #:warmup 0 #:rounds 1 #:load void))))

  (test-case "a child of a checkout without the step hook is timed whole, and says so"
    (call-with-pre-hook-torch
     (lambda (old-torch)
       (define a (run-example fixture #:runner stepper
                              #:settings (list old-torch '("STEPPER_SILENT" . "1"))
                              #:log quiet))
       (check-equal? (hash-ref a 'exit_code) 0)
       (check-equal? (hash-ref a 'timing) "wall")
       (check-equal? (hash-ref a 'steps) 0)
       (check-false (hash-ref a 'error #f))
       (check-true (positive? (hash-ref a 'wall_s))))))

  (test-case "a child of a checkout without the step hook that logs steps is timed by them"
    (call-with-pre-hook-torch
     (lambda (old-torch)
       (define a (run-example fixture #:runner stepper #:settings (list old-torch) #:log quiet))
       (check-equal? (hash-ref a 'timing) "steps")
       (check-equal? (hash-ref a 'steps) 9))))

  (test-case "a run timed whole is measured by its wall time"
    (check-equal? (primary-ms fixture "old" (hasheq 'timing "wall" 'wall_s 2.5 'exit_code 0))
                  2500.0))

  (test-case "a run without steps is timed whole only from a torch known to lack the hook"
    (check-regexp-match #rx"named no torch" (hash-ref (classify '() 5.0 #f) 'error))
    (check-regexp-match #rx"step hook" (hash-ref (classify '() 5.0 torch-dir) 'error))
    (with-temporary-path (old (make-temporary-directory "torch-~a"))
      (check-equal? (classify '() 5.0 old) (hasheq 'steps 0 'timing "wall"))))

  (test-case "a child that exits non-zero after a full run fails"
    (define a (run-example fixture #:runner teardown-crash #:log quiet))
    (check-equal? (hash-ref a 'exit_code) 139)
    (check-equal? (hash-ref a 'steps) 9)
    (check-regexp-match #rx"exited non-zero" (hash-ref a 'error #f))
    (check-exn (lambda (e)
                 (and (exn:fail:user:e2e? e)
                      (equal? (exn:fail:user:e2e-exit-code e) 139)))
               (lambda () (primary-ms fixture #f a))))

  (test-case "a child past its time limit is killed and fails"
    (define t0 (current-inexact-monotonic-milliseconds))
    (define a (run-example fixture #:runner sleeper #:time-limit 5 #:log quiet))
    (check-true (< (- (current-inexact-monotonic-milliseconds) t0) 60000.0))
    (check-regexp-match #rx"no exit within 5 s" (hash-ref a 'error #f))
    (check-not-equal? (hash-ref a 'exit_code) 0))

  (test-case "a child past its time limit is killed with its process group"
    (call-with-pid-file
     (lambda (pid-setting pid-file)
       (define t0 (current-inexact-monotonic-milliseconds))
       (define a (run-example fixture #:runner grandchild #:settings (list pid-setting)
                              #:time-limit 15 #:log quiet))
       (check-true (< (- (current-inexact-monotonic-milliseconds) t0) 60000.0))
       (check-regexp-match #rx"no exit within 15 s" (hash-ref a 'error #f))
       (check-true (within? 10 (lambda () (dead? (grandchild-pid pid-file))))))))

  (define (break-grandchild-run pid-setting pid-file log)
    (define raised (box #f))
    (define running
      (thread (lambda ()
                (with-handlers ([exn:break? (lambda (e) (set-box! raised e))])
                  (run-example fixture #:runner grandchild #:settings (list pid-setting)
                               #:time-limit 600 #:log log)))))
    (check-true (within? 120 (lambda () (file-exists? pid-file))))
    (break-thread running)
    (check-not-false (sync/timeout 30 running))
    (unbox raised))

  (test-case "a break kills the child's process group and is re-raised"
    (call-with-pid-file
     (lambda (pid-setting pid-file)
       (check-pred exn:break? (break-grandchild-run pid-setting pid-file quiet))
       (check-true (within? 10 (lambda () (dead? (grandchild-pid pid-file))))))))

  (test-case "a broken child's output still reaches a log that is not a file"
    (call-with-pid-file
     (lambda (pid-setting pid-file)
       (define log (open-output-string))
       (check-pred exn:break? (break-grandchild-run pid-setting pid-file log))
       (check-regexp-match #rx"grandchild running" (get-output-string log)))))

  (test-case "a child that exits leaving a background job is not held by it, and the job is killed"
    (call-with-pid-file
     (lambda (pid-setting pid-file)
       (define t0 (current-inexact-monotonic-milliseconds))
       (define a (run-example fixture #:runner lingering #:settings (list pid-setting)
                              #:time-limit 60 #:log quiet))
       (check-true (< (- (current-inexact-monotonic-milliseconds) t0) 60000.0))
       (check-equal? (hash-ref a 'exit_code) 0)
       (check-false (hash-ref a 'error #f))
       (check-equal? (hash-ref a 'steps) 9)
       (check-true (within? 10 (lambda () (dead? (grandchild-pid pid-file))))))))

  (test-case "a child that exits non-zero says so beside why it has no time"
    (check-regexp-match #rx"^exited non-zero; .*no measured steps"
                        (hash-ref (run-example fixture #:runner broken #:log quiet) 'error #f)))

  (test-case "a child that leaves a partial steps file is an error, not a raise"
    (define a (run-example fixture #:runner partial-steps #:log quiet))
    (check-equal? (hash-ref a 'exit_code) 1)
    (check-regexp-match #rx"unreadable steps file" (hash-ref a 'error #f)))

  (test-case "a torch installed compiled only still has the step hook"
    (with-temporary-path (compiled-only (make-temporary-directory "torch-~a"))
      (define compiled (build-path compiled-only "private" "compiled"))
      (make-directory* compiled)
      (call-with-output-file (build-path compiled "step-log_rkt.zo") void)
      (check-regexp-match #rx"step hook" (hash-ref (classify '() 5.0 compiled-only) 'error))))

  (test-case "a child sees only the example settings it is given"
    (check-equal? (control-variables (cadr examples) (example-runner (cadr examples)))
                  (append '("STEPS") recorder-variables '("EXCERPT")))
    (check-equal? (control-variables fixture stepper)
                  (append '("EPOCHS") recorder-variables '("STEPPER_SILENT" "STEPS")))
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (putenv "STEPS" "5")
      (putenv "EPOCHS" "1")
      (check-equal? (hash-ref (run-example fixture #:runner stepper #:log quiet) 'steps) 9)
      (define given (run-example fixture #:runner stepper #:settings '(("STEPS" . "7"))
                                 #:log quiet))
      (check-equal? (hash-ref given 'steps) 7)
      (check-equal? (hash-ref (hash-ref given 'settings) 'STEPS) "7")))

  (test-case "the recorder child runs under the collection policy its variant sets, and no other"
    (with-temporary-path (dir (make-temporary-directory "bench-collect-~a"))
      (define (seen-by settings)
        (define out (path->string (build-path dir "seen")))
        (run-example fixture #:runner collect-reader
                     #:settings (cons (cons "BENCH_ENV_OUT" out) settings) #:log quiet)
        (call-with-input-file out read))
      (check-equal? (seen-by '(("RKTORCH_BENCH_COLLECT_BUDGET" . "0.2"))) '(#t 1/5))
      (check-equal? (seen-by '(("RKTORCH_BENCH_COLLECT_AT_TROUGHS" . "0"))) '(#f 1/20))
      (parameterize ([current-environment-variables
                      (environment-variables-copy (current-environment-variables))])
        (putenv "RKTORCH_BENCH_COLLECT_AT_TROUGHS" "0")
        (putenv "RKTORCH_BENCH_COLLECT_BUDGET" "0.5")
        (check-equal? (seen-by '()) '(#t 1/20) "the parent's policy does not leak"))))

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
