#lang racket/base

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
         (only-in racket/cmdline command-line)
         (only-in racket/file make-directory*)
         (only-in racket/format ~a)
         (only-in racket/future processor-count)
         (only-in racket/hash hash-union)
         (only-in racket/list append* shuffle)
         (only-in racket/match match match-define)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-split)
         (only-in threading ~>)
         (only-in torch torch-version)
         (only-in "harness.rkt" comma-list host-meta load-average write-record)
         (only-in "m0-cells.rkt" m0-pipelines)
         (only-in "op-cost.rkt" default-out-dir inside-repo? out-paths stamp)
         (only-in "op-cost-pipelines.rkt" single-threaded-environment)
         (only-in "pytorch.rkt" child-records require-pytorch twin-records)
         (only-in "runtime.rkt" outside-cores process-cpu-times)
         (only-in "session.rkt" fresh-run-id wait-for-load)
         (only-in "shim.rkt" aten-threads)
         (only-in "summary-m0.rkt" m0-markdown)
         (only-in "summary.rkt" read-records))

(provide block-arms
         cell-args
         m0-paths
         m0-plan
         m0-settings
         racket-label
         run-arm
         run-blocks)

(define-runtime-path cells-module "m0-cells.rkt")
(define-runtime-path op-cost-module "op-cost.rkt")
(define-runtime-path repo-root "../..")

(define default-samples
  (hasheq 'cifar-png 2048 'cifar-batched 4096 'jpeg 512 'cifar 4096 'text 4096))

(define smoke-settings
  (hasheq 'rounds 1 'samples 64 'epochs 1 'warmup 1 'seconds 0.5 'dataset "fixture" 'k3-smoke? #t))

(define (block-arms pipeline workers)
  (if (or (zero? workers) (eq? pipeline 'train))
      '(rktorch pytorch)
      '(rktorch pytorch pytorch-persistent)))

(define (m0-plan #:rounds rounds #:pipelines pipelines #:workers workers
                 #:train-workers train-workers #:seed seed)
  (define cells
    (append (for*/list ([p (in-list pipelines)] [w (in-list workers)]) (cons p w))
            (for/list ([w (in-list train-workers)]) (cons 'train w))))
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed seed)
    (append*
     (for/list ([round (in-range rounds)])
       (for/list ([c (in-list (shuffle cells))] [i (in-naturals)])
         (match-define (cons p w) c)
         (hasheq 'round round 'block i 'pipeline p 'workers w
                 'arms (shuffle (block-arms p w))))))))

(define (racket-label requested [running (version)])
  (define major.minor (car (regexp-match #rx"^[0-9]+[.][0-9]+" running)))
  (cond
    [(or (not requested) (equal? requested major.minor)) major.minor]
    [else
     (raise-user-error 'm0 "asked for Racket ~a, but this is Racket ~a: run m0.rkt under ~a's racket"
                       requested running requested)]))

(define (m0-paths dir label [tag (stamp)])
  (define base (format "m0-~a-~a" label tag))
  (values (build-path dir (string-append base ".jsonl"))
          (build-path dir (string-append base ".md"))
          base))

(define (cell-args block arm settings)
  (define p (hash-ref block 'pipeline))
  (define w (hash-ref block 'workers))
  (define (setting key) (hash-ref settings key))
  (append
   (if (eq? arm 'pytorch-persistent) '("--persistent") '())
   (match p
     ['train (list "--seconds" (setting 'seconds) "--warmup" (setting 'warmup) "train" w)]
     [_ (list "--warmup" (setting 'warmup) "--epochs" (setting 'epochs)
              "--samples" (or (setting 'samples) (hash-ref default-samples p 0))
              "--dataset" (setting 'dataset) "loader" p w)])))

(define (child-arm block arm settings #:log log)
  (define args (cell-args block arm settings))
  (define limit (hash-ref settings 'time-limit))
  (match (parameterize ([current-environment-variables (single-threaded-environment)])
           (match arm
             ['rktorch (child-records (find-exe) (cons cells-module args)
                                      #:log log #:time-limit limit #:who 'm0-cell)]
             [_ (twin-records "loaders.py" args #:log log #:time-limit limit)]))
    [(list record) record]
    [records (raise-user-error 'm0 "~a printed ~a records, not one" arm (length records))]))

(define (run-arm block arm order settings
                 #:run [run child-arm]
                 #:cpu-times [cpu-times (lambda () (process-cpu-times #:children? #t))]
                 #:load [load load-average]
                 #:log [log (current-error-port)])
  (define before (cpu-times))
  (define load-start (load))
  (define child
    (with-handlers ([exn:fail? (lambda (e) (hasheq 'error (exn-message e)))])
      (run block arm settings #:log log)))
  (define outside (outside-cores before (cpu-times) (processor-count)))
  (hash-union child
              (hasheq 'kind "arm"
                      'round (hash-ref block 'round)
                      'block (hash-ref block 'block)
                      'order order
                      'pipeline (~a (hash-ref block 'pipeline))
                      'workers (hash-ref block 'workers)
                      'arm (~a arm)
                      'impl (if (eq? arm 'rktorch) "rktorch" "pytorch")
                      'racket (hash-ref settings 'racket)
                      'max_load (hash-ref settings 'max-load)
                      'load load-start
                      'load_end (load)
                      'load_outside outside)
              #:combine/key (lambda (_k _child ours) ours)))

(define (run-blocks plan settings #:emit emit #:log log #:run [run child-arm]
                    #:wait [wait wait-for-load])
  (for ([block (in-list plan)])
    (wait (hash-ref settings 'max-load) log)
    (for ([arm (in-list (hash-ref block 'arms))] [order (in-naturals)])
      (define r (run-arm block arm order settings #:run run #:log log))
      (fprintf log "round ~a block ~a ~a w~a ~a: ~a\n" (hash-ref block 'round)
               (hash-ref block 'block) (hash-ref block 'pipeline) (hash-ref block 'workers) arm
               (or (hash-ref r 'samples_per_s #f) (hash-ref r 'main_steps_per_s #f)
                   (hash-ref r 'error #f)))
      (emit r))))

(define (k3-run dir base settings log)
  (define env (single-threaded-environment))
  (environment-variables-set! env #"BASELINE_DATE" (string->bytes/utf-8 base))
  (define args
    (append (if (hash-ref settings 'k3-smoke?) '("--smoke") '())
            (list "--out-dir" dir "--suites" "ops,pytorch" "--threads" "1,8"
                  "--reps" (hash-ref settings 'k3-reps))))
  (parameterize ([current-environment-variables env])
    (child-records (find-exe) (cons op-cost-module args) #:log log
                   #:time-limit (* 4 (hash-ref settings 'time-limit)) #:who 'm0-op-cost))
  (define-values (jsonl _md) (out-paths dir base))
  jsonl)

(define (op-cost-records records)
  (for*/list ([r (in-list records)]
              #:when (and (equal? (hash-ref r 'kind #f) "op-cost") (file-exists? (hash-ref r 'path))
                          (hash-ref r 'racket #f))
              [k (in-list (read-records (list (hash-ref r 'path))))])
    (hash-set k 'racket (hash-ref r 'racket))))

(define (summarize-files paths)
  (define records (read-records paths))
  (m0-markdown records #:op-cost (op-cost-records records)))

(define (run-m0 settings #:log [log (current-error-port)])
  (define dir (hash-ref settings 'out-dir))
  (when (inside-repo? dir repo-root)
    (raise-user-error 'm0 "~a is inside the repository; bench results are never committed" dir))
  (define aten (aten-threads))
  (unless (eqv? aten 1)
    (raise-user-error 'm0 "needs one ATen intra-op thread, got ~a: set OMP_NUM_THREADS=1" aten))
  (require-pytorch 'm0)
  (define label (hash-ref settings 'racket))
  (define-values (jsonl md base) (m0-paths dir label))
  (when (file-exists? jsonl)
    (raise-user-error 'm0 "~a exists; one records file holds one run (set BASELINE_DATE anew)"
                      jsonl))
  (make-directory* dir)
  (define run-id (fresh-run-id))
  (define plan
    (m0-plan #:rounds (hash-ref settings 'rounds) #:pipelines (hash-ref settings 'pipelines)
             #:workers (hash-ref settings 'workers)
             #:train-workers (hash-ref settings 'train-workers) #:seed (hash-ref settings 'seed)))
  (call-with-output-file jsonl #:exists 'error
    (lambda (out)
      (define (emit r) (write-record (hash-set r 'run_id run-id) out))
      (emit (hasheq 'kind "meta" 'racket label
                    'meta (host-meta #:device 'cpu #:libtorch (torch-version) #:threads aten
                                     #:repo repo-root)
                    'settings (for/hasheq ([(k v) (in-hash settings)])
                                (values k (if (list? v) (map ~a v) (~a v))))))
      (run-blocks plan settings #:emit emit #:log log)
      (when (positive? (hash-ref settings 'k3-reps))
        (emit (with-handlers ([exn:fail? (lambda (e)
                                           (hasheq 'kind "op-cost-error" 'error (exn-message e)))])
                (hasheq 'kind "op-cost" 'racket label
                        'path (path->string (k3-run dir base settings log))))))))
  (call-with-output-file md #:exists 'truncate
    (lambda (out) (write-string (summarize-files (list jsonl)) out)))
  (fprintf log "records: ~a\nsummary: ~a\n" jsonl md))

(define (default-settings)
  (hasheq 'rounds 5 'racket #f 'out-dir (default-out-dir) 'max-load 2.0
          'pipelines m0-pipelines 'workers '(0 1 2 4 7) 'train-workers '(1 2 4 8)
          'samples #f 'warmup 1 'epochs 2 'seconds 3.0 'dataset "full" 'seed 555
          'k3-reps #f 'k3-smoke? #f 'time-limit 1800 'summarize #f))

(define (m0-settings argv)
  (define settings (default-settings))
  (define (put! key value) (set! settings (hash-set settings key value)))
  (define (count text [ok? exact-positive-integer?])
    (define n (string->number text))
    (unless (ok? n)
      (raise-user-error 'm0 "expected a count, got ~s" text))
    n)
  (define (counts text)
    (map (lambda (t) (count t exact-nonnegative-integer?)) (string-split text ",")))
  (define (positive text)
    (define x (string->number text))
    (unless (and (real? x) (< 0 x +inf.0))
      (raise-user-error 'm0 "expected a positive number, got ~s" text))
    x)
  (define summarize
    (command-line
     #:program "m0"
     #:argv argv
     #:once-each
     [("--rounds") n "Rounds, each block once per round (default 5)" (put! 'rounds (count n))]
     [("--racket") v "The Racket version this run must be, e.g. 9.3 or 9.1" (put! 'racket v)]
     [("--out-dir") d "Directory for m0-<racket>-<stamp>.jsonl and .md" (put! 'out-dir d)]
     [("--max-load") l "Wait for fewer busy cores before a block; drop a block above it"
                     (put! 'max-load (positive l))]
     [("--pipelines") s "Comma-separated: cifar-png,cifar-batched,jpeg,cifar,text"
                      (put! 'pipelines (map string->symbol (string-split s ",")))]
     [("--workers") s "Comma-separated loader worker counts (default 0,1,2,4,7)"
                    (put! 'workers (counts s))]
     [("--train-workers") s "Comma-separated #439 worker counts (default 1,2,4,8); none: -"
                          (put! 'train-workers (if (equal? s "-") '() (counts s)))]
     [("--samples") n "Samples per epoch for every pipeline" (put! 'samples (count n))]
     [("--warmup") n "Untimed epochs before the timed ones" (put! 'warmup (count n exact-nonnegative-integer?))]
     [("--epochs") n "Timed epochs per arm" (put! 'epochs (count n))]
     [("--seconds") s "Seconds of a #439 training run" (put! 'seconds (positive s))]
     [("--dataset") d "The CIFAR-10 PNG tree: full or fixture" (put! 'dataset d)]
     [("--seed") n "Seed of the block and arm order" (put! 'seed (count n exact-nonnegative-integer?))]
     [("--k3-reps") n "op-cost.rkt reps for K3 (default: --rounds; 0 skips)"
                    (put! 'k3-reps (count n exact-nonnegative-integer?))]
     [("--time-limit") s "Kill an arm after s seconds (default 1800)" (put! 'time-limit (positive s))]
     [("--smoke") "One round of the smallest arms on the PNG fixture; flags after it override"
                  (set! settings (hash-union settings smoke-settings
                                             #:combine (lambda (_default smoke) smoke)))]
     #:args paths paths))
  (hash-set* settings
             'racket (racket-label (hash-ref settings 'racket))
             'k3-reps (or (hash-ref settings 'k3-reps) (hash-ref settings 'rounds))
             'summarize (and (pair? summarize) summarize)))

(module+ main
  (define settings (m0-settings (current-command-line-arguments)))
  (match (hash-ref settings 'summarize)
    [#f (run-m0 settings)]
    [paths (write-string (summarize-files paths))]))
