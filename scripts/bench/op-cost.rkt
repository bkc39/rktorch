#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/date current-date date->string date-display-format)
         (only-in racket/file make-directory*)
         (only-in racket/match match)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-prefix? string-split string-trim)
         (only-in threading ~>)
         (only-in torch torch-version)
         (only-in "harness.rkt"
                  busy-cores comma-list host-meta load-average proc-field rotate write-record)
         (only-in "op-cost-layers.rkt" pure-overhead-ops)
         (only-in "op-cost-measure.rkt" conditions run-rep sampler-record warm-up!)
         (only-in "op-cost-ops.rkt" ladder-ops op-case-layer op-case-name)
         (only-in "op-cost-pipelines.rkt" pipeline-records single-threaded-environment)
         (only-in "pytorch.rkt" require-pytorch twin-records)
         (only-in "shim.rkt" aten-threads)
         (only-in "summary-op-cost.rkt" op-cost-markdown)
         (only-in "summary.rkt" read-records)
         (only-in "system-tools.rkt" command-output))

(provide cpu-list-count
         cpu-warning
         default-out-dir
         inside-repo?
         out-paths
         smoke-settings)

(define-runtime-path repo-root "../..")

(define suites '(ops pytorch sampler pipelines))

(define (default-out-dir)
  (or (getenv "BASELINE_DIR")
      (~> (or (getenv "XDG_DATA_HOME") (build-path (find-system-path 'home-dir) ".local" "share"))
          (build-path "rktorch" "bench")
          path->string)))

(define (inside-repo? dir [root repo-root])
  (define (dir-string p)
    (~> p path->complete-path simplify-path path->directory-path path->string))
  (string-prefix? (dir-string dir) (dir-string root)))

(define (stamp)
  (or (getenv "BASELINE_DATE")
      (parameterize ([date-display-format 'iso-8601])
        (~> (current-date) (date->string #t) (regexp-replace* #rx":" _ "")))))

(define (out-paths dir [tag (stamp)])
  (values (build-path dir (format "op-cost-~a.jsonl" tag))
          (build-path dir (format "op-cost-~a.md" tag))))

;; One timed epoch of 64 samples, with the warm-up epoch kept: PyTorch's
;; persistent workers start inside the first epoch.
(define smoke-settings
  (hasheq 'reps 1 'seconds 0.05 'thread-seconds 0.1 'pipeline-reps 1
          'warmup 1 'epochs 1 'samples 64))

(define (cpu-list-count text)
  (for/sum ([part (in-list (string-split (string-trim text) ","))])
    (match (map string->number (string-split part "-"))
      [(list lo hi) (add1 (- hi lo))]
      [(list _) 1])))

(define (affinity-cpus)
  (define listed (proc-field "/proc/self/status" "Cpus_allowed_list"))
  (define nproc (and (not listed) (command-output "nproc")))
  (cond
    [listed (cpu-list-count listed)]
    [nproc (string->number nproc)]
    [else #f]))

;; main and the reapers run beside n threads or n workers
(define (cpu-warning cpus #:threads threads #:workers workers)
  (define n (apply max 0 (append threads workers)))
  (and cpus (positive? n) (<= cpus n)
       (format "op-cost: warning: ~a CPUs in the affinity mask for ~a threads or workers; pin ~a"
               cpus n (add1 n))))

(define (twin-rep rep settings log emit)
  (for ([r (in-list (parameterize ([current-environment-variables
                                    (single-threaded-environment)])
                      (twin-records "op_cost.py" (append (list "--rep" rep) settings)
                                    #:log log)))])
    (emit (hash-set r 'rep rep))))

(define sampled-ops (append pure-overhead-ops '(zeros-8x8 narrow)))

(define (sampler-records cases seconds emit)
  (for* ([c (in-list cases)]
         #:when (memq (op-case-name c) sampled-ops)
         [layer (in-list '(facade contracted))]
         #:when (op-case-layer c layer))
    (emit (sampler-record c layer seconds))))

(define (run-scoreboard #:out-dir dir #:suites chosen #:ops ops #:threads threads #:reps reps
                        #:seconds seconds #:thread-seconds thread-seconds
                        #:pipelines pipelines #:pipeline-workers pipeline-workers
                        #:pipeline-reps pipeline-reps #:pipeline-settings p-settings
                        #:log [log (current-error-port)]
                        #:aten-threads [aten (aten-threads)]
                        #:cpus [cpus (affinity-cpus)])
  (when (inside-repo? dir)
    (raise-user-error 'op-cost "~a is inside the repository; bench results are never committed"
                      dir))
  (unless (eqv? aten 1)
    (raise-user-error 'op-cost "needs one ATen intra-op thread, got ~a: set OMP_NUM_THREADS=1"
                      aten))
  (define-values (jsonl md) (out-paths dir))
  (define warning
    (cpu-warning cpus
                 #:threads (if (or (memq 'ops chosen) (memq 'pytorch chosen)) threads '())
                 #:workers (if (memq 'pipelines chosen) pipeline-workers '())))
  (when warning (displayln warning log))
  (when (file-exists? jsonl)
    (raise-user-error 'op-cost "~a exists; one records file holds one run (set BASELINE_DATE anew)"
                      jsonl))
  (when (memq 'pytorch chosen) (require-pytorch 'op-cost))
  (make-directory* dir)
  (call-with-output-file jsonl #:exists 'error
    (lambda (out)
      (define (emit r) (write-record r out))
      (emit (hasheq 'kind "meta"
                    'meta (host-meta #:device 'cpu #:libtorch (torch-version)
                                     #:threads aten #:repo repo-root)
                    'settings (hasheq 'reps reps 'seconds seconds 'thread_seconds thread-seconds
                                      'threads threads 'ops (map symbol->string ops))
                    'load (load-average)))
      (define cases (ladder-ops ops))
      (define conds (conditions cases threads))
      (define twin-settings
        (list "--seconds" seconds "--thread-seconds" thread-seconds
              "--threads" (comma-list threads) "--ops" (comma-list ops)))
      (define sides (filter (lambda (s) (memq s chosen)) '(ops pytorch)))
      (when (memq 'ops chosen) (warm-up! conds))
      (for* ([rep (in-range reps)] [side (in-list (rotate sides rep))])
        (fprintf log "rep ~a ~a load ~a\n" rep side (load-average))
        (emit (hasheq 'kind "load" 'rep rep 'side (symbol->string side)
                      'busy_cores (busy-cores) 'load (load-average)))
        (case side
          [(ops) (run-rep cases conds rep #:seconds seconds #:thread-seconds thread-seconds
                          #:emit emit)]
          [(pytorch) (twin-rep rep twin-settings log emit)]))
      (when (memq 'sampler chosen)
        (sampler-records cases (* 4 seconds) emit))
      (when (memq 'pipelines chosen)
        (pipeline-records pipelines pipeline-workers pipeline-reps
                          #:settings p-settings #:emit emit #:log log))))
  (call-with-output-file md #:exists 'truncate
    (lambda (out) (~> jsonl read-records op-cost-markdown (write-string out))))
  (fprintf log "records: ~a\nsummary: ~a\n" jsonl md))

(module+ main
  (require (only-in json write-json)
           (only-in racket/cmdline command-line)
           (only-in racket/match match-define)
           (only-in "op-cost-ops.rkt" op-cost-names)
           (only-in "op-cost-pipelines.rkt" pipeline-settings run-cell)
           (only-in "threads.rkt" thread-counts)
           (only-in "workloads/pipelines.rkt" pipeline-names))

  (define out-dir (default-out-dir))
  (define chosen suites)
  (define ops op-cost-names)
  (define threads thread-counts)
  (define reps 5)
  (define seconds 0.5)
  (define thread-seconds 1.0)
  (define pipelines pipeline-names)
  (define pipeline-workers '(0 8))
  (define pipeline-reps 3)
  (define warmup 1)
  (define epochs 2)
  (define samples #f)
  (define cell #f)
  (define summary #f)
  (define (symbols s) (map string->symbol (string-split s ",")))
  (define (numbers s) (map string->number (string-split s ",")))
  (define (positive s)
    (define n (string->number s))
    (unless (and (real? n) (positive? n))
      (raise-user-error 'op-cost "expected a positive number, got ~s" s))
    n)
  (command-line
   #:once-each
   [("--out-dir") d "Directory for op-cost-<stamp>.jsonl and .md (default BASELINE_DIR)"
                  (set! out-dir d)]
   [("--suites") s "Comma-separated: ops,pytorch,sampler,pipelines" (set! chosen (symbols s))]
   [("--ops") s "Comma-separated ops" (set! ops (symbols s))]
   [("--threads") s "Comma-separated #:pool 'own thread counts" (set! threads (numbers s))]
   [("--reps") n "Interleaved repetitions" (set! reps (positive n))]
   [("--seconds") s "Seconds per one-thread sample" (set! seconds (positive s))]
   [("--thread-seconds") s "Seconds per threaded sample" (set! thread-seconds (positive s))]
   [("--pipelines") s "Comma-separated pipelines" (set! pipelines (symbols s))]
   [("--pipeline-workers") s "Comma-separated worker counts" (set! pipeline-workers (numbers s))]
   [("--pipeline-reps") n "Repetitions per pipeline cell" (set! pipeline-reps (positive n))]
   [("--warmup") n "Pipeline warm-up epochs" (set! warmup (string->number n))]
   [("--epochs") n "Pipeline timed epochs" (set! epochs (positive n))]
   [("--samples") n "Pipeline samples per epoch (0: the pipeline's own)"
                  (set! samples (let ([v (string->number n)]) (and v (positive? v) v)))]
   [("--cell") name workers "Run one rktorch pipeline cell and print its record"
               (set! cell (list (string->symbol name) (string->number workers)))]
   [("--summarize") p "Print the markdown summary of a records file" (set! summary p)]
   [("--smoke") "Smallest settings, warm-up epoch kept; flags after it override"
                (set!-values (reps seconds thread-seconds pipeline-reps warmup epochs samples)
                             (apply values
                                    (map (lambda (k) (hash-ref smoke-settings k))
                                         '(reps seconds thread-seconds pipeline-reps warmup
                                                epochs samples))))])
  (cond
    [cell
     (match-define (list name workers) cell)
     (write-json (run-cell name workers #:warmup warmup #:epochs epochs #:samples samples))
     (newline)]
    [summary (~> summary read-records op-cost-markdown write-string)]
    [else
     (run-scoreboard #:out-dir out-dir #:suites chosen #:ops ops #:threads threads #:reps reps
                     #:seconds seconds #:thread-seconds thread-seconds
                     #:pipelines pipelines #:pipeline-workers pipeline-workers
                     #:pipeline-reps pipeline-reps
                     #:pipeline-settings (pipeline-settings #:warmup warmup #:epochs epochs
                                                            #:samples samples))]))

(module+ test
  (require (only-in racket/file delete-directory/files file->string make-temporary-directory)
           (only-in racket/port open-output-nowhere)
           rackunit)

  (define (refusal dir #:aten-threads [aten 1] #:log [log (open-output-nowhere)]
                   #:suites [suites '()] #:workers [workers '()] #:cpus [cpus 64])
    (with-handlers ([exn:fail:user? exn-message])
      (run-scoreboard #:out-dir dir #:suites suites #:ops '() #:threads '() #:reps 1
                      #:seconds 0.1 #:thread-seconds 0.1 #:pipelines '()
                      #:pipeline-workers workers #:pipeline-reps 1 #:pipeline-settings '()
                      #:log log #:aten-threads aten #:cpus cpus)
      #f))

  (test-case "an affinity mask of n CPUs or fewer for n threads or workers draws one warning"
    (check-equal? (cpu-list-count "0-8") 9)
    (check-equal? (cpu-list-count "0,2,4-7\n") 6)
    (check-equal? (cpu-warning 2 #:threads '(1 2) #:workers '())
                  "op-cost: warning: 2 CPUs in the affinity mask for 2 threads or workers; pin 3")
    (check-false (cpu-warning 2 #:threads '(1) #:workers '()))
    (check-false (cpu-warning 9 #:threads '(1 2 4 8) #:workers '(0 8)))
    (check-regexp-match #rx"for 8 threads or workers"
                        (cpu-warning 8 #:threads '() #:workers '(0 8)))
    (check-false (cpu-warning #f #:threads '(8) #:workers '()))
    (check-false (cpu-warning 1 #:threads '() #:workers '(0))))

  (test-case "the driver writes the warning for the cells it will run, on one line"
    (define dir (make-temporary-directory))
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (putenv "BASELINE_DATE" "2026-11-11")
      (define-values (jsonl _md) (out-paths dir))
      (call-with-output-file jsonl (lambda (out) (write-string "{}\n" out)))
      (define (logged suites cpus)
        (define log (open-output-string))
        (refusal dir #:suites suites #:workers '(2) #:cpus cpus #:log log)
        (get-output-string log))
      (check-equal? (logged '(pipelines) 2)
                    (string-append (cpu-warning 2 #:threads '() #:workers '(2)) "\n"))
      (check-equal? (logged '(pipelines) 3) "")
      (check-equal? (logged '(ops) 2) ""))
    (delete-directory/files dir))

  (test-case "smoke settings keep the warm-up epoch, so worker start-up is never timed"
    (check-true (>= (hash-ref smoke-settings 'warmup) 1))
    (check-true (>= (hash-ref smoke-settings 'reps) 1)))

  (test-case "a run refuses a directory inside the repository, and more than one ATen thread"
    (check-regexp-match #rx"is inside the repository" (refusal (build-path repo-root "scripts")))
    (define dir (make-temporary-directory))
    (check-regexp-match #rx"needs one ATen intra-op thread, got 4"
                        (refusal (build-path dir "out") #:aten-threads 4))
    (check-false (directory-exists? (build-path dir "out")))
    (delete-directory/files dir))

  (test-case "a run refuses a records file an earlier run wrote, and leaves it as it was"
    (define dir (make-temporary-directory))
    (define earlier "{\"kind\":\"meta\"}\n")
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (putenv "BASELINE_DATE" "2026-11-04")
      (define-values (jsonl _md) (out-paths dir))
      (call-with-output-file jsonl (lambda (out) (write-string earlier out)))
      (check-regexp-match #rx"op-cost-2026-11-04.jsonl exists" (refusal dir))
      (check-equal? (file->string jsonl) earlier))
    (delete-directory/files dir))

  (test-case "results go outside the repository, to BASELINE_DIR when it is set"
    (check-true (inside-repo? (build-path repo-root "scripts" "bench")))
    (check-true (inside-repo? repo-root))
    (define dir (make-temporary-directory))
    (check-false (inside-repo? dir))
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (putenv "BASELINE_DIR" (path->string dir))
      (check-equal? (default-out-dir) (path->string dir)))
    (define-values (jsonl md) (out-paths dir "2026-11-04"))
    (check-equal? jsonl (build-path dir "op-cost-2026-11-04.jsonl"))
    (check-equal? md (build-path dir "op-cost-2026-11-04.md"))
    (delete-directory/files dir)))
