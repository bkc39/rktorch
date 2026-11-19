#lang racket/base

(require (only-in racket/match match match*)
         (only-in torch make-generator)
         (only-in torch/data/loader dataloader)
         (only-in "harness.rkt" clock-ms)
         (only-in "op-cost-pipelines.rkt" epoch! peak-rss-kb)
         (only-in "shim.rkt" aten-threads)
         (only-in "threads.rkt" scaling-run)
         (only-in (submod "threads.rkt" private) settled-thread-count task-dir with-peak-sampler)
         (only-in "workloads/cifar-pipeline.rkt" cifar-folder-loader cifar-folder-root)
         (only-in "workloads/pipelines.rkt" pipeline pipeline-names))

(provide gc-count
         loader-cell
         m0-pipelines
         measured
         png-pipelines
         table-h-pipelines
         train-cell)

(define png-pipelines '(cifar-png cifar-batched))
(define m0-pipelines (append png-pipelines pipeline-names))
(define table-h-pipelines '(cifar-png cifar-batched jpeg))

(define png-batch 128)

(define (gc-count)
  (define stats (make-vector 4 0))
  (vector-set-performance-stats! stats)
  (vector-ref stats 3))

(define (measured thunk #:tasks [tasks task-dir])
  (define baseline (settled-thread-count tasks))
  (define gc0 (current-gc-milliseconds))
  (define count0 (gc-count))
  (define t0 (clock-ms))
  (define-values (result peak) (with-peak-sampler (#:tasks tasks) (thunk)))
  (values result
          (hasheq 'elapsed_s (/ (- (clock-ms) t0) 1000.0)
                  'gc_ms (- (current-gc-milliseconds) gc0)
                  'gc_count (- (gc-count) count0)
                  'os_threads_baseline baseline
                  'os_threads_peak peak)))

(define (m0-loader name workers #:samples samples #:dataset dataset)
  (define (png-loader form)
    (cifar-folder-loader (cifar-folder-root dataset #f) form (make-generator 0)
                         #:batch png-batch #:workers workers #:samples samples))
  (match name
    ['cifar-png (png-loader "per-sample")]
    ['cifar-batched (png-loader "batched")]
    [_
     (define-values (ds batch-size) (pipeline name #:samples samples))
     (dataloader ds #:batch-size batch-size #:shuffle? #t #:workers workers)]))

(define (process-fields stats)
  (hash-set* stats
             'racket (version)
             'vm (symbol->string (system-type 'vm))
             'aten_threads (aten-threads)
             'omp_num_threads (getenv "OMP_NUM_THREADS")
             'mkl_num_threads (getenv "MKL_NUM_THREADS")
             'peak_rss_mb (let ([kb (peak-rss-kb)]) (and kb (quotient kb 1024)))))

(define (loader-cell name workers #:warmup warmup #:epochs epochs #:samples samples
                     #:dataset [dataset "full"])
  (define loader (m0-loader name workers #:samples samples #:dataset dataset))
  (for ([_ (in-range warmup)]) (epoch! loader))
  (define-values (n stats)
    (measured (lambda () (for/sum ([_ (in-range epochs)]) (epoch! loader)))))
  (hash-set* (process-fields stats)
             'samples n
             'samples_per_s (/ n (hash-ref stats 'elapsed_s))
             'warmup warmup
             'epochs epochs))

(define (train-cell workers #:seconds seconds #:warmup warmup)
  (for ([_ (in-range warmup)]) (scaling-run workers 'train #:seconds seconds))
  (define-values (run stats) (measured (lambda () (scaling-run workers 'train #:seconds seconds))))
  (hash-set* (process-fields stats)
             'main_steps_per_s (hash-ref run 'main_steps_per_s)
             'ops_per_s (hash-ref run 'ops_per_s)
             'seconds seconds
             'warmup warmup))

(module+ main
  (require (only-in json write-json)
           (only-in racket/cmdline command-line))

  (define settings (make-hasheq '((warmup . 1) (epochs . 2) (samples . #f) (seconds . 2.0)
                                  (dataset . "full"))))
  (define (put! key value) (hash-set! settings key value))
  (define (setting key) (hash-ref settings key))
  (define-values (kind args)
    (command-line
     #:once-each
     [("--warmup") n "Untimed epochs, or training runs" (put! 'warmup (string->number n))]
     [("--epochs") n "Timed epochs" (put! 'epochs (string->number n))]
     [("--samples") n "Samples per epoch" (put! 'samples (string->number n))]
     [("--seconds") s "Seconds of a training run" (put! 'seconds (string->number s))]
     [("--dataset") d "The CIFAR-10 PNG tree: full or fixture" (put! 'dataset d)]
     #:args (kind . args) (values kind args)))
  (write-json
   (match* (kind args)
     [("loader" (list name workers))
      (loader-cell (string->symbol name) (string->number workers)
                   #:warmup (setting 'warmup) #:epochs (setting 'epochs)
                   #:samples (setting 'samples) #:dataset (setting 'dataset))]
     [("train" (list workers))
      (train-cell (string->number workers) #:seconds (setting 'seconds)
                  #:warmup (setting 'warmup))]))
  (newline))
