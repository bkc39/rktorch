#lang racket/base

;; Diagnostic (#556 R6/R7, not for merge): M0's loader cells under the
;; tensor-scope arms against their PyTorch twins (python/loaders.py).
;;   racket scope-pipelines.rkt [--out-dir DIR] [--pipelines cifar-png,cifar,jpeg]
;;     [--workers 0,4,7] [--arms off,build,ref] [--rounds 3] [--samples N]
;;     [--warmup 1] [--epochs 2] [--dataset full|fixture] [--no-pytorch]
;; runs every arm of every cell in a fresh process (so peak_rss_mb is that
;; arm's VmHWM), appends one JSON line per run to DIR/scope-pipelines-<stamp>.jsonl
;; and prints the median table; --summarize FILE... prints the table of earlier
;; runs; --cell PIPELINE WORKERS runs one rktorch cell under the arm in
;; RKTORCH_DIAG_SCOPE and writes its record.

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
         (only-in json write-json)
         (only-in racket/cmdline command-line)
         (only-in racket/file make-directory*)
         (only-in racket/list remove-duplicates shuffle)
         (only-in racket/math exact-round)
         (only-in racket/runtime-path define-runtime-path)
         (only-in racket/string string-join string-split)
         (only-in torch/foreign/raw/collector finalizer-runs)
         (only-in torch/foreign/raw/scope scope-diagnostics)
         (only-in "cifar-folder.rkt" ensure-cifar10-folder!)
         (only-in "harness.rkt" load-average write-record)
         (only-in "m0-cells.rkt" loader-cell png-pipelines)
         (only-in "op-cost.rkt" default-out-dir inside-repo? stamp)
         (only-in "op-cost-pipelines.rkt" single-threaded-environment)
         (only-in "pytorch.rkt" child-records require-pytorch twin-records)
         (only-in "summary.rkt" read-records))

(define-runtime-path self "scope-pipelines.rkt")
(define-runtime-path repo-root "../..")

(define default-samples (hasheq 'cifar-png 2048 'cifar-batched 4096 'jpeg 512 'cifar 4096 'text 4096))

(define (arm-from-env)
  (or (getenv "RKTORCH_DIAG_SCOPE") "off"))

(define (run-cell name workers settings)
  (define scope0 (scope-diagnostics))
  (define runs0 (finalizer-runs))
  (define record
    (loader-cell name workers #:warmup (hash-ref settings 'warmup)
                 #:epochs (hash-ref settings 'epochs) #:samples (hash-ref settings 'samples)
                 #:dataset (hash-ref settings 'dataset)))
  (define scope1 (scope-diagnostics))
  (for/fold ([r (hash-set* record 'arm (arm-from-env)
                           'finalizer_runs (- (finalizer-runs) runs0))])
            ([(k v) (in-hash scope1)])
    (hash-set r (string->symbol (format "scope_~a" k)) (- v (hash-ref scope0 k)))))

(define (cell-args name workers settings)
  (list "--warmup" (hash-ref settings 'warmup) "--epochs" (hash-ref settings 'epochs)
        "--samples" (or (hash-ref settings 'samples) (hash-ref default-samples name))
        "--dataset" (hash-ref settings 'dataset)))

(define (child-env arm)
  (define env (single-threaded-environment))
  (environment-variables-set! env #"RKTORCH_DIAG_SCOPE" (string->bytes/utf-8 (symbol->string arm)))
  env)

(define (run-side side name workers settings log)
  (define args (cell-args name workers settings))
  (define limit (hash-ref settings 'time-limit))
  (define records
    (with-handlers ([exn:fail? (lambda (e) (list (hasheq 'error (exn-message e))))])
      (case side
        [(pytorch)
         (parameterize ([current-environment-variables (single-threaded-environment)])
           (twin-records "loaders.py" (append args (list "loader" name workers))
                         #:log log #:time-limit limit))]
        [else
         (child-records (find-exe)
                        (append (list self "--cell" name workers) args)
                        #:environment (child-env side) #:log log #:time-limit limit
                        #:who 'scope-pipelines)])))
  (if (= (length records) 1)
      (car records)
      (hasheq 'error (format "~a printed ~a records, not one" side (length records)))))

(define (show v)
  (cond
    [(not v) "-"]
    [(and (real? v) (>= (abs v) 100)) (number->string (exact-round v))]
    [(real? v) (real->decimal-string v 2)]
    [else (format "~a" v)]))

(define (median xs)
  (define sorted (sort xs <))
  (and (pair? sorted) (list-ref sorted (quotient (length sorted) 2))))

(define (cell-runs runs pipeline workers side)
  (for/list ([r (in-list runs)]
             #:when (and (equal? (hash-ref r 'pipeline #f) pipeline)
                         (equal? (hash-ref r 'workers #f) workers)
                         (equal? (hash-ref r 'side #f) side)
                         (not (hash-ref r 'error #f))))
    r))

(define (field-median rs k)
  (median (filter real? (map (lambda (r) (hash-ref r k #f)) rs))))

(define (summarize runs [out (current-output-port)])
  (define (distinct key) (remove-duplicates (map (lambda (r) (hash-ref r key #f)) runs)))
  (define sides (filter values (distinct 'side)))
  (fprintf out "\n| pipeline | workers | side | n | samples/s | /PyTorch | GC ms | peak RSS MB | exit-frees | escaped-to-reaper | finalizer runs |\n")
  (fprintf out "|---|---|---|---|---|---|---|---|---|---|---|\n")
  (for* ([pipeline (in-list (filter values (distinct 'pipeline)))]
         [workers (in-list (sort (filter values (distinct 'workers)) <))]
         [side (in-list sides)])
    (define rs (cell-runs runs pipeline workers side))
    (define py (field-median (cell-runs runs pipeline workers "pytorch") 'samples_per_s))
    (define rate (field-median rs 'samples_per_s))
    (when (pair? rs)
      (fprintf out "| ~a | ~a | ~a | ~a | ~a | ~a | ~a | ~a | ~a | ~a | ~a |\n"
               pipeline workers side (length rs) (show rate)
               (if (and py rate) (real->decimal-string (/ rate py) 2) "-")
               (show (field-median rs 'gc_ms)) (show (field-median rs 'peak_rss_mb))
               (show (field-median rs 'scope_exit-frees))
               (show (field-median rs 'scope_escaped-to-reaper))
               (show (field-median rs 'finalizer_runs)))))
  (for ([r (in-list runs)] #:when (hash-ref r 'error #f))
    (fprintf out "\nerror ~a w=~a ~a round ~a:\n~a\n" (hash-ref r 'pipeline #f)
             (hash-ref r 'workers #f) (hash-ref r 'side #f) (hash-ref r 'round #f)
             (hash-ref r 'error))))

(define (drive settings #:log [log (current-error-port)])
  (define dir (hash-ref settings 'out-dir))
  (when (inside-repo? dir repo-root)
    (raise-user-error 'scope-pipelines "~a is inside the repository; bench results are never committed" dir))
  (define pipelines (hash-ref settings 'pipelines))
  (when (for/or ([p (in-list pipelines)]) (memq p png-pipelines))
    (ensure-cifar10-folder! (hash-ref settings 'dataset)))
  (define sides (append (if (hash-ref settings 'pytorch?) '(pytorch) '()) (hash-ref settings 'arms)))
  (when (memq 'pytorch sides) (require-pytorch 'scope-pipelines))
  (make-directory* dir)
  (define path (build-path dir (format "scope-pipelines-~a.jsonl" (stamp))))
  (define runs
    (call-with-output-file path #:exists 'error
      (lambda (out)
        (for*/list ([round (in-range (hash-ref settings 'rounds))]
                    [cell (in-list (shuffle (for*/list ([p (in-list pipelines)]
                                                        [w (in-list (hash-ref settings 'workers))])
                                              (cons p w))))]
                    [side (in-list (shuffle sides))])
          (define name (car cell))
          (define workers (cdr cell))
          (define load (load-average))
          (define record
            (hash-set* (run-side side name workers settings log)
                       'pipeline (symbol->string name) 'workers workers
                       'side (symbol->string side) 'round round 'load load))
          (write-record record out)
          (fprintf log "round ~a ~a w~a ~a: ~a samples/s, GC ~a ms, peak RSS ~a MB~a\n" round name
                   workers side (show (hash-ref record 'samples_per_s #f))
                   (show (hash-ref record 'gc_ms #f)) (show (hash-ref record 'peak_rss_mb #f))
                   (if (hash-ref record 'error #f) (format " ERROR ~a" (hash-ref record 'error)) ""))
          record))))
  (summarize runs)
  (fprintf log "records: ~a\n" path))

(module+ main
  (define settings
    (make-hasheq `((out-dir . ,(default-out-dir)) (pipelines . (cifar-png cifar jpeg))
                   (workers . (0 4 7)) (arms . (off build ref)) (rounds . 3) (samples . #f)
                   (warmup . 1) (epochs . 2) (dataset . "full") (pytorch? . #t)
                   (time-limit . 1800))))
  (define (put! k v) (hash-set! settings k v))
  (define (symbols s) (map string->symbol (string-split s ",")))
  (define cell #f)
  (define summary '())
  (command-line
   #:once-each
   [("--out-dir") d "Directory of the records file (outside the repository)" (put! 'out-dir d)]
   [("--pipelines") p "Comma-separated M0 pipelines: cifar-png,cifar-batched,jpeg,cifar,text"
                    (put! 'pipelines (symbols p))]
   [("--workers") w "Comma-separated worker counts" (put! 'workers (map string->number (string-split w ",")))]
   [("--arms") a "Comma-separated arms: off,build,ref" (put! 'arms (symbols a))]
   [("--rounds") n "Rounds, every arm of every cell once per round" (put! 'rounds (string->number n))]
   [("--samples") n "Samples per epoch (default: M0's per pipeline)" (put! 'samples (string->number n))]
   [("--warmup") n "Untimed epochs" (put! 'warmup (string->number n))]
   [("--epochs") n "Timed epochs" (put! 'epochs (string->number n))]
   [("--dataset") d "The CIFAR-10 PNG tree: full or fixture" (put! 'dataset d)]
   [("--time-limit") s "Kill a run after s seconds" (put! 'time-limit (string->number s))]
   [("--no-pytorch") "Skip the PyTorch twin" (put! 'pytorch? #f)]
   [("--summarize") "Print the table of the records files given as arguments" (set! summary #t)]
   [("--cell") name workers "Run one rktorch cell in this process"
               (set! cell (list (string->symbol name) (string->number workers)))]
   #:args paths
   (when (and summary (pair? paths)) (set! summary paths)))
  (cond
    [cell
     (write-json (run-cell (car cell) (cadr cell) settings))
     (newline)]
    [(pair? summary) (summarize (read-records summary))]
    [else (drive settings)]))
