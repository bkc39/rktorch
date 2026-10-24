#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in torch accelerator-if-available device-type)
         (only-in "child.rkt" python-runner?)
         (only-in "e2e.rkt"
                  cache-env e2e-cases example-epochs-var example-name example-runner
                  exn:fail:user:e2e merged-settings runner-reads settings->jsexpr)
         (only-in "harness.rkt"
                  host-meta load-average mean measure result result->record result-per-call
                  summarize)
         (only-in "pytorch.rkt" merged-meta)
         (only-in "session.rkt" fresh-run-id record-writer wait-for-load)
         (only-in "shim.rkt" aten-threads))

(provide run-e2e
         paired-groups)

(define (pair-key ex)
  (regexp-replace #rx"^py-(.*?)(-per-param)?$" (example-name ex) "\\1"))

(define (paired-groups exs)
  (for/list ([k (in-list (remove-duplicates (map pair-key exs)))])
    (filter (lambda (ex) (equal? (pair-key ex) k)) exs)))

(define (run-e2e exs #:rounds rounds #:full? full? #:variants variants #:cache cache
                 #:max-load max-load #:libtorch libtorch #:out out #:log log
                 #:run-id [run-id (fresh-run-id)]
                 #:time-limit [time-limit #f]
                 #:runner-for [runner-for example-runner])
  (define emit (record-writer run-id out))
  (define device (device-type (accelerator-if-available)))
  (define scale (if full? 'full 'short))
  (define cached (cache-env cache))
  (define with-cache
    (for/list ([v (in-list variants)])
      (match-define (cons name settings) v)
      (cons name (append (filter (lambda (c) (not (assoc (car c) settings))) cached) settings))))
  (define (impl-of ex) (if (python-runner? (runner-for ex)) "pytorch" "rktorch"))
  (for ([group (in-list (paired-groups exs))])
    (wait-for-load max-load log)
    (define meta (host-meta #:device device #:libtorch libtorch #:threads (aten-threads)))
    (define kept (make-hash))
    (define metas (make-hash))
    (define (push! field key value)
      (when value (hash-update! kept (cons field key) (lambda (xs) (cons value xs)) '())))
    (define (pushed field key)
      (reverse (hash-ref kept (cons field key) '())))
    (define (meta-for impl child-meta)
      (if (equal? impl "pytorch") (merged-meta meta child-meta) (or child-meta meta)))
    (define (on-run run-of variant analysis)
      (define impl (impl-of run-of))
      (define key (cons (example-name run-of) variant))
      (define child-meta (hash-ref analysis 'meta #f))
      (define run-meta (meta-for impl child-meta))
      (fprintf log "~a~a: ~a steps/s, epochs ~a s\n" (example-name run-of)
               (if variant (format "/~a" variant) "")
               (hash-ref analysis 'steps_per_s #f) (hash-ref analysis 'epoch_s '()))
      (emit (hash-set* analysis 'schema 1 'suite "e2e" 'group "run" 'impl impl
                       'case (example-name run-of) 'variant variant
                       'load (load-average)
                       'meta run-meta))
      (cond
        [(hash-ref analysis 'error #f) (void)]
        [else
         (push! 'rates key (hash-ref analysis 'steps_per_s #f))
         (push! 'gcs key (hash-ref analysis 'gc_ms #f))
         (push! 'epoch-gcs key (mean (hash-ref analysis 'epoch_gc_ms '())))
         (push! 'step-mss key (hash-ref (hash-ref analysis 'step_ms (hasheq)) 'median #f))
         (push! 'walls key (hash-ref analysis 'wall_s #f))
         (push! 'timings key (hash-ref analysis 'timing))
         (hash-ref! metas key run-meta)]))
    (define (on-failure e)
      (match-define (list case variant exit-code child-meta)
        (match e
          [(exn:fail:user:e2e _ _ case variant exit-code child-meta)
           (list case variant exit-code child-meta)]
          [(? exn:fail:user?) (list (example-name (car group)) #f #f #f)]
          [_ (raise e)]))
      (define impl (impl-of (for/first ([ex (in-list group)]
                                        #:when (equal? (example-name ex) case))
                              ex)))
      (fprintf log "~a\n" (exn-message e))
      (emit (hasheq 'schema 1 'suite "e2e" 'group "error"
                    'impl impl
                    'case case
                    'variant variant
                    'exit_code exit-code
                    'error (exn-message e)
                    'meta (meta-for impl child-meta)
                    'meta_from (if child-meta "child" "parent"))))
    (define results
      (measure (e2e-cases group #:scale scale #:variants with-cache #:on-run on-run
                          #:time-limit time-limit #:log log #:runner-for runner-for)
               #:warmup 0 #:rounds rounds #:on-failure on-failure))
    (define (by-wall? ex)
      (for/or ([v (in-list with-cache)])
        (and (member "wall" (pushed 'timings (cons (example-name ex) (car v)))) #t)))
    (for* ([(r ex+v) (in-parallel (in-list results)
                                  (in-list (for*/list ([ex (in-list group)] [v (in-list with-cache)])
                                             (cons ex v))))]
           [ex (in-value (car ex+v))]
           [v (in-value (cdr ex+v))]
           [key (in-value (cons (example-name ex) (car v)))]
           [wall? (in-value (by-wall? ex))]
           [timed (in-value (if wall?
                                (struct-copy result r
                                             [per-call (map (lambda (s) (* 1000.0 s))
                                                            (pushed 'walls key))])
                                r))]
           #:when (pair? (result-per-call timed)))
      (define-values (timing unit unit-scale)
        (cond
          [wall? (values "wall" "s/run" 1e-3)]
          [(example-epochs-var ex) (values "steps" "s/epoch" 1e-3)]
          [else (values "steps" "ms/step" 1)]))
      (define rates-of (pushed 'rates key))
      (emit
       (result->record timed #:suite "e2e" #:group "summary" #:unit unit
                       #:meta (hash-ref metas key meta)
                       #:scale unit-scale
                       #:extra (hasheq 'case (example-name ex)
                                       'impl (impl-of ex)
                                       'variant (car v)
                                       'timing timing
                                       'settings (settings->jsexpr
                                                  (merged-settings ex scale (cdr v)))
                                       'reads (runner-reads ex (runner-for ex))
                                       'gc_ms (pushed 'gcs key)
                                       'epoch_gc_ms (pushed 'epoch-gcs key)
                                       'step_ms (pushed 'step-mss key)
                                       'steps_per_s (and (pair? rates-of)
                                                         (summarize rates-of))))))))

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file make-temporary-directory make-temporary-file)
           (only-in racket/list remove-duplicates)
           (only-in racket/port open-output-nowhere)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "e2e.rkt" example find-examples)
           (only-in "fixtures/pre-hook-torch.rkt" call-with-pre-hook-torch)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "pytorch.rkt" pytorch-available?)
           (only-in "summary.rkt" read-records summary-markdown))

  (define-runtime-path broken "fixtures/broken.rkt")
  (define-runtime-path cache-reader "fixtures/cache-reader.rkt")
  (define-runtime-path meta-then-fail "fixtures/meta_then_fail.py")
  (define-runtime-path sleeper "fixtures/sleeper.rkt")
  (define-runtime-path stepper "fixtures/stepper.rkt")
  (define-runtime-path teardown-crash "fixtures/teardown-crash.rkt")
  (define quiet (open-output-nowhere))

  (define (records-of thunk)
    (with-temporary-path (path (make-temporary-file "bench-~a.jsonl"))
      (call-with-output-file path #:exists 'truncate thunk)
      (read-records (list path))))

  (test-case "an e2e run records every child, a summary per variant, and a failure"
    (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0 'first))
    (define (run variants rounds out)
      (run-e2e (list fixture) #:rounds rounds #:full? #f #:variants variants #:cache #f
               #:max-load #f #:libtorch "x" #:out out #:log quiet
               #:runner-for (lambda (_ex) stepper)))
    (define records
      (records-of (lambda (out)
                    (run '(("a") ("b" ("EPOCHS" . "2"))) 2 out)
                    (run '((#f ("EPOCHS" . "1"))) 1 out))))
    (define (group g) (filter (lambda (r) (equal? (hash-ref r 'group) g)) records))
    (match-define (list _ second-run)
      (remove-duplicates (map (lambda (r) (hash-ref r 'run_id)) records)))
    (check-equal? (map (lambda (r) (hash-ref r 'run_id)) (group "error")) (list second-run))
    (check-equal? (map (lambda (r) (hash-ref r 'steps)) (group "run")) '(9 6 6 9 3))
    (check-regexp-match #rx"no measured steps" (hash-ref (car (group "error")) 'error))
    (define summaries (group "summary"))
    (check-equal? (map (lambda (r) (hash-ref r 'variant)) summaries) '("a" "b"))
    (check-equal? (map (lambda (r) (hash-ref (hash-ref r 'settings) 'EPOCHS)) summaries)
                  '("3" "2"))
    (check-equal? (map (lambda (r) (hash-ref (hash-ref r 'stats) 'n)) summaries) '(2 2))
    (check-equal? (map (lambda (r) (list (hash-ref r 'timing) (hash-ref r 'unit))) summaries)
                  '(("steps" "s/epoch") ("steps" "s/epoch")))
    (define md (summary-markdown records))
    (check-regexp-match #rx"## End to end" md)
    (check-regexp-match #rx"\\| stepper \\| b vs a \\|" md)
    (check-regexp-match #rx"n < 6" md)
    (check-regexp-match #rx"## Failures" md)
    (check-regexp-match
     #rx"\\| stepper \\| - \\| `[^`]*`[^|]* \\| [^|]* \\| 0 \\| e2e: stepper: no measured" md))

  (test-case "an example and its twins share their rounds, in rotation"
    (define (stepper-named name)
      (example name '(("EPOCHS" . 3)) '(("EPOCHS" . 3)) "EPOCHS" 0 'first))
    (check-equal? (map (lambda (g) (map example-name g))
                       (paired-groups (find-examples '("05" "09" "py-05" "py-05-mnist-per"))))
                  '(("05-mnist" "py-05-mnist" "py-05-mnist-per-param") ("09-resnet")))
    (define records
      (records-of (lambda (out)
                    (run-e2e (list (stepper-named "stepper") (stepper-named "py-stepper"))
                             #:rounds 2 #:full? #f #:variants '((#f)) #:cache #f #:max-load #f
                             #:libtorch "x" #:out out #:log quiet
                             #:runner-for (lambda (_ex) stepper)))))
    (check-equal? (for/list ([r (in-list records)] #:when (equal? (hash-ref r 'group) "run"))
                    (hash-ref r 'case))
                  '("stepper" "py-stepper" "py-stepper" "stepper"))
    (check-equal? (for/list ([r (in-list records)] #:when (equal? (hash-ref r 'group) "summary"))
                    (list (hash-ref r 'case) (hash-ref (hash-ref r 'stats) 'n)))
                  '(("stepper" 2) ("py-stepper" 2))))

  (test-case "an A/B against a checkout without the step hook compares whole runs"
    (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0 'first))
    (call-with-pre-hook-torch
     (lambda (old-torch)
       (define records
         (records-of (lambda (out)
                       (run-e2e (list fixture) #:rounds 2 #:full? #f
                                #:variants `(("a") ("old" ,old-torch ("STEPPER_SILENT" . "1")))
                                #:cache #f #:max-load #f #:libtorch "x" #:out out #:log quiet
                                #:runner-for (lambda (_ex) stepper)))))
       (define (group g) (filter (lambda (r) (equal? (hash-ref r 'group) g)) records))
       (define runs (group "run"))
       (check-equal? (map (lambda (r) (hash-ref r 'timing)) runs) '("steps" "wall" "wall" "steps"))
       (check-equal? (group "error") '())
       (define summaries (group "summary"))
       (check-equal? (for/list ([r (in-list summaries)])
                       (list (hash-ref r 'variant) (hash-ref r 'timing) (hash-ref r 'unit)))
                     '(("a" "wall" "s/run") ("old" "wall" "s/run")))
       (for ([s (in-list summaries)])
         (define walls
           (for/list ([r (in-list runs)]
                      #:when (equal? (hash-ref r 'variant) (hash-ref s 'variant)))
             (hash-ref r 'wall_s)))
         (check-equal? (length (hash-ref s 'samples)) 2)
         (for ([sample (in-list (hash-ref s 'samples))] [wall (in-list walls)])
           (check-= sample wall 1e-9)))
       (check-false (hash-ref (cadr summaries) 'steps_per_s))
       (check-true (hash? (hash-ref (car summaries) 'steps_per_s)))
       (check-regexp-match #rx"\\| stepper \\| old vs a \\|" (summary-markdown records)))))

  (test-case "a child of a checkout without the step hook that crashes is no whole-run time"
    (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 3)) "EPOCHS" 0 'first))
    (call-with-pre-hook-torch
     (lambda (old-torch)
       (define records
         (records-of (lambda (out)
                       (run-e2e (list fixture) #:rounds 1 #:full? #f
                                #:variants `(("old" ,old-torch ("STEPPER_SILENT" . "1")))
                                #:cache #f #:max-load #f #:libtorch "x" #:out out #:log quiet
                                #:runner-for (lambda (_ex) teardown-crash)))))
       (match-define (list run failure) records)
       (check-equal? (map (lambda (r) (hash-ref r 'group)) records) '("run" "error"))
       (check-equal? (hash-ref run 'exit_code) 139)
       (check-regexp-match #rx"exited non-zero" (hash-ref run 'error #f))
       (check-equal? (hash-ref failure 'exit_code) 139))))

  (test-case "a time limit reaches every e2e child"
    (define fixture (example "sleeper" '() '() #f 0 'first))
    (define t0 (current-inexact-monotonic-milliseconds))
    (define records
      (records-of (lambda (out)
                    (run-e2e (list fixture) #:rounds 1 #:full? #f #:variants '((#f))
                             #:cache #f #:max-load #f #:libtorch "x" #:out out #:log quiet
                             #:time-limit 5 #:runner-for (lambda (_ex) sleeper)))))
    (check-true (< (- (current-inexact-monotonic-milliseconds) t0) 60000.0))
    (check-equal? (map (lambda (r) (hash-ref r 'group)) records) '("run" "error"))
    (for ([r (in-list records)])
      (check-regexp-match #rx"no exit within 5 s" (hash-ref r 'error))))

  (test-case "a failed variant is not run again, and the others run every round"
    (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0 'first))
    (define records
      (records-of (lambda (out)
                    (run-e2e (list fixture) #:rounds 2 #:full? #f
                             #:variants '(("a") ("b" ("EPOCHS" . "1"))) #:cache #f
                             #:max-load #f #:libtorch "x" #:out out #:log quiet
                             #:runner-for (lambda (_ex) stepper)))))
    (check-equal? (map (lambda (r) (list (hash-ref r 'group) (hash-ref r 'variant))) records)
                  '(("run" "a") ("run" "b") ("error" "b") ("run" "a") ("summary" "a")))
    (match-define (list a-run _ _ a-again a-summary) records)
    (check-equal? (hash-ref (hash-ref a-summary 'stats) 'n) 2)
    (check-= (hash-ref (hash-ref a-summary 'steps_per_s) 'median)
             (/ (+ (hash-ref a-run 'steps_per_s) (hash-ref a-again 'steps_per_s)) 2.0)
             1e-9)
    (define md (summary-markdown records))
    (check-regexp-match #rx"## End to end[^#]*\\| stepper \\| a \\|[^#]*## Failures" md)
    (check-regexp-match #rx"\\| stepper \\| b \\| `[^`]*`[^|]* \\| [a-z]+, [0-9]+ threads " md))

  (test-case "with a cache, the summary records the settings its runs used"
    (define fixture (example "stepper" '(("EPOCHS" . 2)) '(("EPOCHS" . 2)) "EPOCHS" 0 'first))
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([name (in-list (environment-variables-names env))]
          #:when (regexp-match? #rx#"^RKTORCH_.*_DIR$" name))
      (environment-variables-set! env name #f))
    (define records
      (parameterize ([current-environment-variables env])
        (records-of (lambda (out)
                      (run-e2e (list fixture) #:rounds 1 #:full? #f #:variants '(("a"))
                               #:cache "c" #:max-load #f #:libtorch "x" #:out out #:log quiet
                               #:runner-for (lambda (_ex) stepper))))))
    (match-define (list run summary)
      (filter (lambda (r) (member (hash-ref r 'group) '("run" "summary"))) records))
    (check-equal? (hash-ref summary 'settings) (hash-ref run 'settings))
    (check-true (for/or ([k (in-hash-keys (hash-ref summary 'settings))])
                  (regexp-match? #rx"^RKTORCH_.*_DIR$" (symbol->string k)))))

  (test-case "a cache root overrides inherited caches, and a variant overrides the cache"
    (define fixture (example "stepper" '(("EPOCHS" . 2)) '(("EPOCHS" . 2)) "EPOCHS" 0 'first))
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"RKTORCH_MNIST_DIR" #"/inherited/mnist")
    (environment-variables-set! env #"RKTORCH_CIFAR10_DIR" #"/inherited/cifar10")
    (define (cache-dir name) (path->string (build-path (current-directory) "c" name)))
    (with-temporary-path (dir (make-temporary-directory "bench-cache-~a"))
      (define (seen-by v) (path->string (build-path dir v)))
      (define records
        (parameterize ([current-environment-variables env])
          (records-of
           (lambda (out)
             (run-e2e (list fixture) #:rounds 1 #:full? #f
                      #:variants `(("a" ("BENCH_ENV_OUT" . ,(seen-by "a")))
                                   ("b" ("BENCH_ENV_OUT" . ,(seen-by "b"))
                                        ("RKTORCH_CIFAR10_DIR" . "/variant/cifar10")))
                      #:cache "c" #:max-load #f #:libtorch "x" #:out out #:log quiet
                      #:runner-for (lambda (_ex) cache-reader))))))
      (define expected
        (hash "a" (list (cache-dir "mnist") (cache-dir "cifar10"))
              "b" (list (cache-dir "mnist") "/variant/cifar10")))
      (for ([v (in-list '("a" "b"))])
        (check-equal? (call-with-input-file (seen-by v) read) (hash-ref expected v) v))
      (check-equal? (map (lambda (r) (hash-ref r 'group)) records)
                    '("run" "run" "summary" "summary"))
      (for ([r (in-list records)])
        (define settings (hash-ref r 'settings))
        (check-equal? (list (hash-ref settings 'RKTORCH_MNIST_DIR)
                            (hash-ref settings 'RKTORCH_CIFAR10_DIR))
                      (hash-ref expected (hash-ref r 'variant))))))

  (test-case "a child that fails before its first step records its own meta"
    (define fixture (example "stepper" '(("EPOCHS" . 1)) '(("EPOCHS" . 1)) "EPOCHS" 0 'first))
    (match-define (list run failure)
      (records-of (lambda (out)
                    (run-e2e (list fixture) #:rounds 1 #:full? #f #:variants '((#f))
                             #:cache #f #:max-load #f #:libtorch "x" #:out out #:log quiet
                             #:runner-for (lambda (_ex) broken)))))
    (check-equal? (map (lambda (r) (hash-ref r 'group)) (list run failure)) '("run" "error"))
    (check-not-equal? (hash-ref failure 'exit_code) 0)
    (check-equal? (hash-ref failure 'meta_from) "child")
    (check-not-equal? (hash-ref (hash-ref failure 'meta) 'libtorch) "x"))

  (cond
    [(not (find-executable-path "python3"))
     (displayln "[e2e-suite] the failing-twin case skipped: no python3")]
    [else
     (test-case "a PyTorch twin that fails after writing its meta keeps the parent's under it"
       (define fixture (example "py-stepper" '(("EPOCHS" . 1)) '(("EPOCHS" . 1)) "EPOCHS" 0 'first))
       (define failure
         (findf (lambda (r) (equal? (hash-ref r 'group) "error"))
                (records-of (lambda (out)
                              (run-e2e (list fixture) #:rounds 1 #:full? #f #:variants '((#f))
                                       #:cache #f #:max-load #f #:libtorch "x" #:out out
                                       #:log quiet #:time-limit 120
                                       #:runner-for (lambda (_ex) meta-then-fail))))))
       (define meta (hash-ref failure 'meta))
       (check-equal? (hash-ref failure 'impl) "pytorch")
       (check-equal? (hash-ref failure 'exit_code) 3)
       (check-equal? (hash-ref failure 'meta_from) "child")
       (check-equal? (hash-ref meta 'python) "3")
       (check-equal? (hash-ref meta 'libtorch #f) "x" "the parent's meta is under the child's"))])

  (test-case "the loader workload and the PyTorch twins run under e2e, and C pairs them"
    (define tiny '(("DATASET" . "fixture") ("EPOCHS" . "2") ("BATCH" . "128") ("BASE" . "4")
                   ("WORKERS" . "2")))
    (define names (if (pytorch-available?)
                      '("17-workers" "py-05" "py-05-mnist-per" "py-09" "py-17" "py-17-workers-per")
                      '("17-workers")))
    (unless (pytorch-available?)
      (displayln "[e2e-suite] PyTorch twins skipped: python3 `torch` not available"))
    (define records
      (records-of (lambda (out)
                    (run-e2e (find-examples names) #:rounds 1 #:full? #f
                             #:variants `(("crop" ("VARIANT" . "crop") ,@tiny) ("pgm" ,@tiny))
                             #:cache #f #:max-load #f #:libtorch "x" #:out out #:log quiet
                             #:time-limit 900))))
    (define (group g) (filter (lambda (r) (equal? (hash-ref r 'group) g)) records))
    (check-equal? (group "error") '())
    (define summaries (group "summary"))
    (check-equal? (length summaries) (* 2 (length names)))
    (for ([r (in-list summaries)])
      (define meta (hash-ref r 'meta))
      (check-equal? (hash-ref r 'impl)
                    (if (regexp-match? #rx"^py-" (hash-ref r 'case)) "pytorch" "rktorch"))
      (check-equal? (hash-ref r 'unit) "s/epoch")
      (check-true (string? (hash-ref meta 'git_sha)) (hash-ref r 'case))
      (check-equal? (length (hash-ref r 'step_ms)) 1)
      (when (equal? (hash-ref r 'impl) "pytorch")
        (define per-param? (regexp-match? #rx"-per-param$" (hash-ref r 'case)))
        (check-equal? (hash-ref meta 'optimizer) (if per-param? "per-param" "default"))
        (check-true (hash-ref meta 'set_to_none))
        (when per-param? (check-false (hash-ref meta 'foreach)))))
    (when (pytorch-available?)
      (define md (summary-markdown records))
      (check-regexp-match #rx"## C\\. Loader end to end[^#]*\\| 17-workers \\| crop \\| s/epoch \\|" md)
      (check-regexp-match #rx"\\| 17-workers \\| pgm \\| s/epoch \\|" md))))
