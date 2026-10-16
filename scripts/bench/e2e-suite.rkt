#lang racket/base

(require (only-in racket/match match match-define)
         (only-in torch accelerator-if-available device-type)
         (only-in "e2e.rkt"
                  cache-env e2e-cases example-epochs-var example-name example-runner
                  exn:fail:user:e2e merged-settings settings->jsexpr)
         (only-in "harness.rkt"
                  host-meta load-average measure result result->record result-per-call summarize)
         (only-in "session.rkt" fresh-run-id record-writer wait-for-load)
         (only-in "shim.rkt" aten-threads))

(provide run-e2e)

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
  (for ([ex (in-list exs)])
    (wait-for-load max-load log)
    (define meta (host-meta #:device device #:libtorch libtorch #:threads (aten-threads)))
    (define rates (make-hash))
    (define gcs (make-hash))
    (define metas (make-hash))
    (define walls (make-hash))
    (define timings (make-hash))
    (define (on-run run-of variant analysis)
      (define run-meta (or (hash-ref analysis 'meta #f) meta))
      (fprintf log "~a~a: ~a steps/s, epochs ~a s\n" (example-name run-of)
               (if variant (format "/~a" variant) "")
               (hash-ref analysis 'steps_per_s #f) (hash-ref analysis 'epoch_s '()))
      (emit (hash-set* analysis 'schema 1 'suite "e2e" 'group "run"
                       'case (example-name run-of) 'variant variant
                       'load (load-average)
                       'meta run-meta))
      (define (push! table value)
        (hash-update! table variant (lambda (xs) (if value (cons value xs) xs)) '()))
      (cond
        [(hash-ref analysis 'error #f) (void)]
        [else
         (push! rates (hash-ref analysis 'steps_per_s #f))
         (push! gcs (hash-ref analysis 'gc_ms #f))
         (push! walls (hash-ref analysis 'wall_s #f))
         (push! timings (hash-ref analysis 'timing))
         (hash-ref! metas variant run-meta)]))
    (define (on-failure e)
      (match-define (list variant exit-code child-meta)
        (match e
          [(exn:fail:user:e2e _ _ variant exit-code child-meta) (list variant exit-code child-meta)]
          [(? exn:fail:user?) (list #f #f #f)]
          [_ (raise e)]))
      (fprintf log "~a\n" (exn-message e))
      (emit (hasheq 'schema 1 'suite "e2e" 'group "error"
                    'case (example-name ex)
                    'variant variant
                    'exit_code exit-code
                    'error (exn-message e)
                    'meta (or child-meta meta)
                    'meta_from (if child-meta "child" "parent"))))
    (define results
      (measure (e2e-cases (list ex) #:scale scale #:variants with-cache #:on-run on-run
                          #:time-limit time-limit #:log log #:runner-for runner-for)
               #:warmup 0 #:rounds rounds #:on-failure on-failure))
    (define by-wall?
      (for/or ([ts (in-hash-values timings)]) (and (member "wall" ts) #t)))
    (define-values (timing unit unit-scale)
      (cond
        [by-wall? (values "wall" "s/run" 1e-3)]
        [(example-epochs-var ex) (values "steps" "s/epoch" 1e-3)]
        [else (values "steps" "ms/step" 1)]))
    (define (variant-list table v)
      (reverse (hash-ref table (car v) '())))
    (define (timed-result r v)
      (if by-wall?
          (struct-copy result r [per-call (map (lambda (s) (* 1000.0 s)) (variant-list walls v))])
          r))
    (for* ([(r v) (in-parallel (in-list results) (in-list with-cache))]
           [timed (in-value (timed-result r v))]
           #:when (pair? (result-per-call timed)))
      (define rates-of (variant-list rates v))
      (emit
       (result->record timed #:suite "e2e" #:group "summary" #:unit unit
                       #:meta (hash-ref metas (car v) meta)
                       #:scale unit-scale
                       #:extra (hasheq 'case (example-name ex)
                                       'variant (car v)
                                       'timing timing
                                       'settings (settings->jsexpr
                                                  (merged-settings ex scale (cdr v)))
                                       'gc_ms (variant-list gcs v)
                                       'steps_per_s (and (pair? rates-of)
                                                         (summarize rates-of))))))))

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file make-temporary-directory make-temporary-file)
           (only-in racket/list remove-duplicates)
           (only-in racket/port open-output-nowhere)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "e2e.rkt" example)
           (only-in "fixtures/pre-hook-torch.rkt" call-with-pre-hook-torch)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "summary.rkt" read-records summary-markdown))

  (define-runtime-path broken "fixtures/broken.rkt")
  (define-runtime-path cache-reader "fixtures/cache-reader.rkt")
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

  (test-case "a failed variant keeps the summaries of the runs that completed"
    (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0 'first))
    (define records
      (records-of (lambda (out)
                    (run-e2e (list fixture) #:rounds 2 #:full? #f
                             #:variants '(("a") ("b" ("EPOCHS" . "1"))) #:cache #f
                             #:max-load #f #:libtorch "x" #:out out #:log quiet
                             #:runner-for (lambda (_ex) stepper)))))
    (check-equal? (map (lambda (r) (list (hash-ref r 'group) (hash-ref r 'variant))) records)
                  '(("run" "a") ("run" "b") ("error" "b") ("summary" "a")))
    (match-define (list a-run _ _ a-summary) records)
    (check-equal? (hash-ref (hash-ref a-summary 'stats) 'n) 1)
    (check-equal? (hash-ref (hash-ref a-summary 'steps_per_s) 'median)
                  (exact->inexact (hash-ref a-run 'steps_per_s)))
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
    (check-not-equal? (hash-ref (hash-ref failure 'meta) 'libtorch) "x")))
