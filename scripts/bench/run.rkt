#lang racket/base

(require (only-in racket/cmdline command-line)
         (only-in racket/format ~a)
         (only-in racket/list check-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/os getpid)
         (only-in racket/string string-split)
         (only-in torch accelerator-if-available device-type torch-version)
         (only-in "contracts.rkt" contract-workloads pipeline-case)
         (only-in "crossings.rkt" crossing-cases)
         (only-in "e2e.rkt"
                  cache-env e2e-cases example-epochs-var example-name example-runner
                  examples exn:fail:user:e2e find-examples merged-settings settings->jsexpr)
         (only-in "harness.rkt"
                  busy-cores format-result host-meta load-average measure result
                  result->record result-name result-per-call summarize write-record)
         (only-in "layers.rkt" layer-cases)
         (only-in "ops.rkt" op-cases)
         (only-in "shim.rkt" aten-threads))

(provide run-suites
         parse-arguments
         parse-variant
         micro-suites)

(define micro-suites '(crossings layers ops contracts))
(define known-suites (append micro-suites '(e2e)))

(define (check-suites suites)
  (define unknown (for/first ([s (in-list suites)] #:unless (memq s known-suites)) s))
  (when unknown
    (raise-user-error 'run "unknown suite ~a; known: micro ~a" unknown known-suites)))

(define (parse-variant text)
  (match-define (list _ name settings) (regexp-match #rx"^([^:]*)(?::(.*))?$" text))
  (cons name
        (for/list ([kv (in-list (if settings (string-split settings ",") '()))])
          (match kv
            [(regexp #rx"^([^=]+)=(.*)$" (list _ var value)) (cons var value)]
            [_ (raise-user-error 'run "a variant setting is VAR=VALUE, got ~s" kv)]))))

(define (flag-value flag parse ok? expected text)
  (match (parse text)
    [(? ok? v) v]
    [_ (raise-user-error 'run "~a expects ~a, got ~s" flag expected text)]))

(define (finite-positive-real? x)
  (and (real? x) (< 0 x +inf.0)))

(define (ops-device? d)
  (and (memq d '(cpu cuda mps)) #t))

(define (fresh-run-id)
  (format "~a-~a-~a" (current-milliseconds) (getpid) (random 1000000000)))

(define ((record-writer run-id out) record)
  (write-record (hash-set record 'run_id run-id) out))

(define (current-load)
  (or (busy-cores)
      (match (load-average)
        [(cons one-minute _) one-minute]
        [_ #f])))

(define (wait-for-load limit log)
  (let loop ()
    (define load (and limit (current-load)))
    (cond
      [(or (not load) (< load limit)) (void)]
      [else
       (fprintf log "load ~a, waiting for < ~a\n" load limit)
       (sleep 10)
       (loop)])))

(define (run-suites suites
                    #:rounds [rounds #f]
                    #:warmup [warmup 1]
                    #:scale [scale 1]
                    #:device [device 'cpu]
                    #:full? [full? #f]
                    #:only [only #f]
                    #:variants [variants '((#f))]
                    #:cache [cache #f]
                    #:max-load [max-load #f]
                    #:run-id [run-id (fresh-run-id)]
                    #:out [out (current-output-port)]
                    #:log [log (current-error-port)])
  (check-suites suites)
  (define libtorch (torch-version))
  (define emit (record-writer run-id out))
  (define (n reps) (max 1 (inexact->exact (round (* reps scale)))))
  (define twice (check-duplicates (map car variants)))
  (when twice
    (raise-user-error 'run "two variants are named ~a" twice))
  (define (measure-group group cases #:unit [unit "ns"] #:scale [unit-scale 1e6]
                         #:case-name [case-name #f] #:on-device [on-device 'cpu])
    (wait-for-load max-load log)
    (define meta (host-meta #:device on-device #:libtorch libtorch #:threads (aten-threads)))
    (fprintf log "~a~a\n" group (if case-name (format " ~a" case-name) ""))
    (for ([r (in-list (measure cases #:warmup warmup #:rounds (or rounds 5)))])
      (fprintf log "  ~a\n" (format-result r #:unit unit #:scale unit-scale))
      (emit
       (result->record r #:suite "micro" #:group (~a group) #:unit unit #:meta meta
                       #:scale unit-scale
                       #:extra (if case-name
                                   (hasheq 'case (~a case-name) 'variant (~a (result-name r)))
                                   (hasheq))))))
  (for ([suite (in-list suites)])
    (case suite
      [(crossings) (measure-group 'crossings (crossing-cases #:reps (n 200000)))]
      [(layers) (measure-group 'layers (layer-cases #:reps (n 20000) #:drain (n 2000)))]
      [(ops)
       (for ([o (in-list (op-cases #:device device #:scale scale))])
         (measure-group 'ops (cdr o) #:case-name (car o) #:on-device device))]
      [(contracts)
       (for ([w (in-list (contract-workloads #:scale scale))])
         (measure-group 'contracts (cdr w) #:case-name (car w)))
       (measure-group 'pipeline (list (pipeline-case #:reps (n 1))) #:unit "ms" #:scale 1)]
      [(e2e)
       (run-e2e (if only (find-examples only) examples)
                #:rounds (or rounds 1) #:full? full? #:variants variants #:cache cache
                #:max-load max-load #:libtorch libtorch #:run-id run-id #:out out #:log log)])))

(define (run-e2e exs #:rounds rounds #:full? full? #:variants variants #:cache cache
                 #:max-load max-load #:libtorch libtorch #:out out #:log log
                 #:run-id [run-id (fresh-run-id)]
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
                          #:log log #:runner-for runner-for)
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

(define (parse-arguments argv)
  (define options (make-hasheq))
  (define (option! key value) (hash-set! options key value))
  (define names
    (let parse ([argv argv] [names '()])
      (command-line
       #:program "run.rkt"
       #:argv argv
       #:usage-help "Suites: micro crossings layers ops contracts e2e."
                    "Options go before or after the suites."
       #:once-each
       [("--rounds") n "Timed rounds (default 5; e2e 1)"
                     (option! 'rounds
                              (flag-value "--rounds" string->number exact-positive-integer?
                                          "a positive integer" n))]
       [("--warmup") n "Untimed rounds before them (default 1; e2e runs none)"
                     (option! 'warmup
                              (flag-value "--warmup" string->number exact-nonnegative-integer?
                                          "a non-negative integer" n))]
       [("--scale") x "Multiply micro repetitions by x"
                    (option! 'scale
                             (flag-value "--scale" string->number finite-positive-real?
                                         "a finite positive number" x))]
       [("--device") d "Device for the ops suite: cpu, cuda or mps"
                     (option! 'device
                              (flag-value "--device" string->symbol ops-device? "cpu, cuda or mps"
                                          d))]
       [("--full") "Run examples at their full settings" (option! 'full? #t)]
       [("--only") names "Comma-separated example numbers"
                   (option! 'only (string-split names ","))]
       [("--cache") dir "Shared dataset cache root for the examples" (option! 'cache dir)]
       [("--max-load") l "Wait for a load below l (busy cores over 1 s on Linux)"
                       (option! 'max-load
                                (flag-value "--max-load" string->number finite-positive-real?
                                            "a finite positive number" l))]
       [("--out") file "Append JSON lines to file" (option! 'out file)]
       #:multi
       [("--variant") v "name:VAR=VALUE,... (repeat for A/B)"
                      (hash-update! options 'variants
                                    (lambda (vs) (append vs (list (parse-variant v))))
                                    '())]
       #:args suite
       (match suite
         ['() (reverse names)]
         [(cons name more) (parse more (cons name names))]))))
  (define suites
    (for*/list ([s (in-list names)]
                [x (in-list (if (equal? s "micro") (map symbol->string micro-suites) (list s)))])
      (string->symbol x)))
  (check-suites suites)
  (values suites options))

(module+ main
  (define-values (suites options)
    (parse-arguments (current-command-line-arguments)))
  (define (option key [default #f]) (hash-ref options key default))
  (define (go out)
    (run-suites suites
                #:rounds (option 'rounds)
                #:warmup (option 'warmup 1) #:scale (option 'scale 1)
                #:device (option 'device 'cpu) #:full? (option 'full?) #:only (option 'only)
                #:variants (option 'variants '((#f)))
                #:cache (option 'cache) #:max-load (option 'max-load) #:out out))
  (match (option 'out)
    [#f (go (current-output-port))]
    [path (call-with-output-file path #:exists 'append go)]))

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file make-temporary-directory make-temporary-file)
           (only-in racket/list remove-duplicates)
           (only-in racket/port open-output-nowhere)
           (only-in racket/runtime-path define-runtime-path)
           rackunit
           (only-in "e2e.rkt" example)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "summary.rkt" read-records summary-markdown))

  (define-runtime-path stepper "fixtures/stepper.rkt")
  (define-runtime-path broken "fixtures/broken.rkt")
  (define-runtime-path cache-reader "fixtures/cache-reader.rkt")
  (define quiet (open-output-nowhere))

  (define (records-of thunk)
    (with-temporary-path (path (make-temporary-file "bench-~a.jsonl"))
      (call-with-output-file path #:exists 'truncate thunk)
      (read-records (list path))))

  (test-case "a variant names its settings"
    (check-equal? (parse-variant "base") '("base"))
    (check-equal? (parse-variant "one:OMP_NUM_THREADS=1,X=a=b")
                  '("one" ("OMP_NUM_THREADS" . "1") ("X" . "a=b")))
    (check-equal? (parse-variant "alt:LD_LIBRARY_PATH=/a:/b")
                  '("alt" ("LD_LIBRARY_PATH" . "/a:/b")))
    (check-exn #rx"VAR=VALUE" (lambda () (parse-variant "bad:X"))))

  (test-case "every micro suite runs, records and summarises"
    (define records
      (records-of (lambda (out)
                    (run-suites micro-suites #:rounds 1 #:warmup 0 #:scale 0.001
                                #:out out #:log quiet))))
    (define groups (for/list ([r (in-list records)]) (hash-ref r 'group)))
    (for ([g (in-list '("crossings" "layers" "ops" "contracts" "pipeline"))])
      (check-not-false (member g groups) g))
    (for ([r (in-list records)] #:unless (equal? (hash-ref r 'case) "finalization"))
      (check-true (positive? (hash-ref (hash-ref r 'stats) 'median)) (hash-ref r 'case)))
    (define md (summary-markdown records))
    (for ([heading (in-list '("## Native crossings" "## The op stack" "## Per op"
                              "## Validation strategies" "## A real pipeline"))])
      (check-regexp-match (regexp-quote heading) md))
    (check-exn #rx"unknown suite nope"
               (lambda () (run-suites '(nope) #:log quiet)))
    (check-exn #rx"two variants are named a"
               (lambda () (run-suites '(e2e) #:variants '(("a") ("a" ("X" . "1")))
                                      #:log quiet))))

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
    (define records
      (records-of (lambda (out)
                    (run-e2e (list fixture) #:rounds 2 #:full? #f
                             #:variants '(("a") ("old" ("BENCH_PREHOOK" . "1"))) #:cache #f
                             #:max-load #f #:libtorch "x" #:out out #:log quiet
                             #:runner-for (lambda (_ex) stepper)))))
    (define (group g) (filter (lambda (r) (equal? (hash-ref r 'group) g)) records))
    (define runs (group "run"))
    (check-equal? (map (lambda (r) (hash-ref r 'timing)) runs) '("steps" "wall" "wall" "steps"))
    (define summaries (group "summary"))
    (check-equal? (for/list ([r (in-list summaries)])
                    (list (hash-ref r 'variant) (hash-ref r 'timing) (hash-ref r 'unit)))
                  '(("a" "wall" "s/run") ("old" "wall" "s/run")))
    (for ([s (in-list summaries)])
      (define walls
        (for/list ([r (in-list runs)] #:when (equal? (hash-ref r 'variant) (hash-ref s 'variant)))
          (hash-ref r 'wall_s)))
      (check-equal? (length (hash-ref s 'samples)) 2)
      (for ([sample (in-list (hash-ref s 'samples))] [wall (in-list walls)])
        (check-= sample wall 1e-9)))
    (check-false (hash-ref (cadr summaries) 'steps_per_s))
    (check-true (hash? (hash-ref (car summaries) 'steps_per_s)))
    (check-equal? (group "error") '())
    (check-regexp-match #rx"\\| stepper \\| old vs a \\|" (summary-markdown records)))

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
    (check-not-equal? (hash-ref (hash-ref failure 'meta) 'libtorch) "x"))

  (test-case "flags go before or after the suites"
    (define (parsed . argv)
      (define-values (suites options) (parse-arguments (list->vector argv)))
      (list suites options))
    (check-equal? (parsed "micro" "--out" "r.jsonl" "--max-load" "3")
                  (list micro-suites (make-hasheq '((out . "r.jsonl") (max-load . 3)))))
    (check-equal? (parsed "--max-load" "3" "ops" "--device" "cuda" "--out" "r.jsonl")
                  (list '(ops) (make-hasheq '((max-load . 3) (device . cuda) (out . "r.jsonl")))))
    (check-equal? (parsed "e2e" "--cache" "c" "--variant" "a" "--out" "r.jsonl"
                          "--variant" "b:X=1")
                  (list '(e2e) (make-hasheq '((cache . "c") (out . "r.jsonl")
                                              (variants . (("a") ("b" ("X" . "1"))))))))
    (check-equal? (parsed "crossings" "--rounds" "2" "layers")
                  (list '(crossings layers) (make-hasheq '((rounds . 2))))))

  (test-case "a flag rejects what is not its value"
    (define (fails flag value expected)
      (check-exn (regexp (string-append (regexp-quote flag) " expects " expected ", got \""
                                        (regexp-quote value) "\""))
                 (lambda () (parse-arguments (vector "micro" flag value)))))
    (fails "--rounds" "six" "a positive integer")
    (fails "--rounds" "0" "a positive integer")
    (fails "--rounds" "1.5" "a positive integer")
    (fails "--warmup" "-1" "a non-negative integer")
    (fails "--warmup" "one" "a non-negative integer")
    (for* ([flag (in-list '("--scale" "--max-load"))]
           [value (in-list '("big" "0" "-1" "1+2i" "+inf.0" "-inf.0" "+nan.0"))])
      (fails flag value "a finite positive number"))
    (fails "--device" "gpu" "cpu, cuda or mps")
    (fails "--device" "CUDA" "cpu, cuda or mps")
    (define-values (_suites options)
      (parse-arguments (vector "micro" "--warmup" "0" "--scale" "0.5" "--max-load" "2.5")))
    (check-equal? options (make-hasheq '((warmup . 0) (scale . 0.5) (max-load . 2.5))))
    (for ([d (in-list '("cpu" "cuda" "mps"))])
      (define-values (_ops parsed) (parse-arguments (vector "ops" "--device" d)))
      (check-equal? (hash-ref parsed 'device) (string->symbol d))))

  (test-case "an unknown suite fails before any suite runs"
    (check-exn #rx"unknown suite --out"
               (lambda () (parse-arguments (vector "micro" "--" "--out" "r.jsonl"))))
    (check-exn #rx"unknown suite nope"
               (lambda () (parse-arguments (vector "micro" "nope" "--out" "r.jsonl"))))
    (check-equal? (records-of (lambda (out)
                                (check-exn #rx"unknown suite nope"
                                           (lambda ()
                                             (run-suites '(crossings nope) #:rounds 1 #:warmup 0
                                                         #:scale 0.001 #:out out #:log quiet)))))
                  '())))
