#lang racket/base

(require (only-in racket/list last remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/os getpid)
         (only-in racket/promise force)
         (only-in threading ~>)
         (only-in torch make-generator shape sum)
         (only-in torch/data/loader
                  dataloader dataset-batch dataset-length dataset-ref default-collate
                  define-dataset)
         (only-in (submod torch/data/loader private) traversal)
         (only-in torch/foreign/raw/accounts ledger-entries)
         (only-in torch/foreign/raw/cas cas-add!)
         (only-in torch/foreign/raw/collector finalizer-runs)
         (only-in torch/foreign/raw/fault with-crossing-hook)
         (only-in (submod torch/runtime/reaper private) reaper-statistics)
         (only-in torch/tests/private/race-harness with-race-hook)
         (only-in "m1-clocks.rkt"
                  call-with-gc-log gc-event-cpu-ms gc-event-end gc-event-live gc-event-mode
                  gc-event-start gc-overlap-ms os-thread-id span-cpu₀ span-cpu₁ span-kind
                  span-tid span-wall₀ span-wall₁ spans task-cpu-ms thread-cpu-ms timed)
         (only-in "runtime-work.rkt" with-trip-bytes)
         (only-in (submod "threads.rkt" private) collector-sweepers)
         (only-in "workloads/cifar-pipeline.rkt" cifar-folder-dataset cifar-folder-root)
         (only-in "workloads/pipelines.rkt" pipeline))

(provide cell-pipelines
         handoff-cell
         span-split)

(define cell-pipelines '(cifar-png jpeg))

(define png-batch 128)

(define-dataset spanned (inner)
  #:length (dataset-length inner)
  #:ref (i)
  (dataset-ref inner i)
  #:batch (indices collate)
  (timed 'build (lambda () (dataset-batch inner indices collate))))

(define ((timed-collate collate) items)
  (timed 'collate (lambda () (collate items))))

(define (cell-source name #:samples samples #:dataset dataset)
  (match name
    ['cifar-png
     (define-values (ds collate)
       (cifar-folder-dataset (cifar-folder-root dataset #f) "per-sample" #:samples samples))
     (values ds collate png-batch)]
    [_
     (define-values (ds batch) (pipeline name #:samples samples))
     (values ds default-collate batch)]))

(define (cell-loader name workers #:samples samples #:dataset dataset)
  (define-values (ds collate batch) (cell-source name #:samples samples #:dataset dataset))
  (dataloader (spanned ds) #:batch-size batch #:shuffle? #t #:workers workers
              #:collate (timed-collate collate) #:generator (make-generator 0)))

(define (epoch! loader)
  (timed 'epoch
         (lambda ()
           (define-values (count batch finish!) (timed 'start (lambda () (traversal loader))))
           (begin0
             (for/sum ([k (in-range count)])
               (match-define (cons x _)
                 (timed 'next (lambda () (call-with-values (lambda () (batch k)) list))))
               (timed 'step (lambda () (void (sum x))))
               (~> x shape car))
             (timed 'finish finish!)))))

(define (counted-epoch! loader)
  (define registered (box 0))
  (define charged (box 0))
  (define crossings (box 0))
  (define n
    (with-race-hook (lambda (name _subject)
                      (case name
                        [(handle-registered) (cas-add! registered 1)]
                        [(ledger-entry-added) (cas-add! charged 1)]
                        [else (void)]))
      (with-crossing-hook (lambda () (cas-add! crossings 1))
        (epoch! loader))))
  (hasheq 'tensors_per_sample (/ (unbox registered) n 1.0)
          'charges_per_sample (/ (unbox charged) n 1.0)
          'crossings_per_sample (/ (unbox crossings) n 1.0)))

(define (sum-of f xs) (for/sum ([x (in-list xs)]) (f x)))

(define (merge-ms a b)
  (for/hasheq ([(k v) (in-hash a)]) (values k (+ v (hash-ref b k)))))

(define (one-worker-idle bs e₀ e₁)
  (define sorted (sort bs < #:key span-wall₀))
  (hasheq 'head_ms (~> (car sorted) span-wall₀ (- e₀))
          'tail_ms (- e₁ (span-wall₁ (last sorted)))
          'gaps_ms (for/sum ([a (in-list sorted)] [b (in-list (cdr sorted))])
                     (- (span-wall₀ b) (span-wall₁ a)))
          'absent_ms 0.0))

(define (worker-idle epochs builds workers)
  (for/fold ([acc (hasheq 'head_ms 0.0 'tail_ms 0.0 'gaps_ms 0.0 'absent_ms 0.0)])
            ([e (in-list epochs)] #:when (positive? workers))
    (define e₀ (span-wall₀ e))
    (define e₁ (span-wall₁ e))
    (define inside (filter (lambda (b) (<= e₀ (span-wall₀ b) (span-wall₁ b) e₁)) builds))
    (define tids (remove-duplicates (map span-tid inside)))
    (define absent (* (max 0 (- workers (length tids))) (- e₁ e₀)))
    (for/fold ([acc (merge-ms acc (hasheq 'head_ms 0.0 'tail_ms 0.0 'gaps_ms 0.0 'absent_ms absent))])
              ([t (in-list tids)])
      (merge-ms acc (one-worker-idle (filter (lambda (b) (eqv? (span-tid b) t)) inside) e₀ e₁)))))

(define (span-split all events #:main main #:workers [workers 0])
  (define (of kind [where (lambda (_s) #t)])
    (filter (lambda (s) (and (eq? (span-kind s) kind) (where s))) all))
  (define (on-main s) (eqv? (span-tid s) main))
  (define (off-main s) (not (on-main s)))
  (define (wall s) (- (span-wall₁ s) (span-wall₀ s)))
  (define (cpu s) (if (and (span-cpu₀ s) (span-cpu₁ s)) (- (span-cpu₁ s) (span-cpu₀ s)) 0.0))
  (define (gc s) (gc-overlap-ms events (span-wall₀ s) (span-wall₁ s)))
  (define (totals ss)
    (hasheq 'wall_ms (sum-of wall ss) 'cpu_ms (sum-of cpu ss) 'gc_ms (sum-of gc ss)
            'count (length ss)))
  (define builds (of 'build off-main))
  (hasheq 'epoch (totals (of 'epoch))
          'start (totals (of 'start))
          'finish (totals (of 'finish))
          'next (totals (of 'next))
          'step (totals (of 'step))
          'main_build (totals (of 'build on-main))
          'main_collate (totals (of 'collate on-main))
          'worker_build (totals builds)
          'worker_collate (totals (of 'collate off-main))
          'worker_threads (~> (map span-tid builds) remove-duplicates length)
          'worker_idle (worker-idle (of 'epoch) builds workers)))

(define (gc-summary events seconds)
  (define (mode-of m) (filter (lambda (e) (eq? (gc-event-mode e) m)) events))
  (define (real e) (- (gc-event-end e) (gc-event-start e)))
  (define (mean xs) (and (pair? xs) (/ (apply + xs) (length xs) 1.0)))
  (define (row es)
    (hasheq 'count (length es)
            'real_ms (sum-of real es)
            'cpu_ms (sum-of gc-event-cpu-ms es)
            'mean_real_ms (mean (map real es))
            'mean_live (mean (filter real? (map gc-event-live es)))))
  (hasheq 'all (row events)
          'minor (row (mode-of 'minor))
          'major (row (mode-of 'major))
          'incremental (row (mode-of 'incremental))
          'per_s (/ (length events) seconds)
          'points (for/list ([e (in-list events)])
                    (list (symbol->string (gc-event-mode e)) (real e) (gc-event-live e)))))

(define (cpu-by-task before after main workers-cpu)
  (define (moved tid) (- (hash-ref after tid 0.0) (hash-ref before tid 0.0)))
  (define persistent (filter (lambda (t) (hash-has-key? after t)) (hash-keys before)))
  (hasheq 'main_ms (moved main)
          'other_persistent_ms (for/sum ([t (in-list persistent)] #:unless (= t main)) (moved t))
          'workers_span_ms workers-cpu))

(define (handoff-cell name workers #:warmup warmup #:epochs epochs #:samples samples
                      #:dataset [dataset "full"] #:trip [trip #f] #:count? [count? #t])
  (force collector-sweepers)
  (define loader (cell-loader name workers #:samples samples #:dataset dataset))
  (define main (os-thread-id))
  (with-trip-bytes trip
    (for ([_ (in-range warmup)]) (epoch! loader))
    (spans #:clear? #t)
    (define tasks₀ (task-cpu-ms))
    (define process₀ (current-process-milliseconds))
    (define gc₀ (current-gc-milliseconds))
    (define runs₀ (finalizer-runs))
    (define reaper₀ (reaper-statistics))
    (define main-cpu₀ (thread-cpu-ms))
    (define-values (n events)
      (call-with-gc-log (lambda () (for/sum ([_ (in-range epochs)]) (epoch! loader)))
                        #:live ledger-entries))
    (define main-cpu₁ (thread-cpu-ms))
    (define all (spans #:clear? #t))
    (define tasks₁ (task-cpu-ms))
    (define split (span-split all events #:main main #:workers workers))
    (define elapsed-ms (hash-ref (hash-ref split 'epoch) 'wall_ms))
    (define reaper₁ (reaper-statistics))
    (hash-set*
     (hasheq 'pipeline (symbol->string name)
             'workers workers
             'samples n
             'elapsed_s (/ elapsed-ms 1000.0)
             'samples_per_s (/ n (/ elapsed-ms 1000.0))
             'us_per_sample (/ (* 1000.0 elapsed-ms) n)
             'trip_mb trip
             'split split
             'gc (gc-summary events (/ elapsed-ms 1000.0))
             'gc_ms (- (current-gc-milliseconds) gc₀)
             'process_cpu_ms (- (current-process-milliseconds) process₀)
             'tasks (cpu-by-task tasks₀ tasks₁ (getpid)
                                 (hash-ref (hash-ref split 'worker_build) 'cpu_ms))
             'main_tid main
             'main_cpu_ms (and main-cpu₀ main-cpu₁ (- main-cpu₁ main-cpu₀))
             'finalizer_runs (- (finalizer-runs) runs₀)
             'reaper_released (- (hash-ref reaper₁ 'released) (hash-ref reaper₀ 'released))
             'ledger_entries_end (ledger-entries)
             'racket (version)
             'warmup warmup
             'epochs epochs)
     'counts (and count? (counted-epoch! loader)))))

(module+ main
  (require (only-in json write-json)
           (only-in racket/cmdline command-line))

  (define settings (make-hasheq '((warmup . 1) (epochs . 2) (samples . #f) (dataset . "full")
                                  (trip . #f))))
  (define (put! key value) (hash-set! settings key value))
  (define (setting key) (hash-ref settings key))
  (define-values (name workers)
    (command-line
     #:once-each
     [("--warmup") n "Untimed epochs" (put! 'warmup (string->number n))]
     [("--epochs") n "Timed epochs" (put! 'epochs (string->number n))]
     [("--samples") n "Samples per epoch" (put! 'samples (string->number n))]
     [("--dataset") d "The CIFAR-10 PNG tree: full or fixture" (put! 'dataset d)]
     [("--trip") mb "collect-trip-bytes in MB while the cell runs (default: Racket's)"
                 (put! 'trip (string->number mb))]
     #:args (_loader name workers)
     (values (string->symbol name) (string->number workers))))
  (unless (memq name cell-pipelines)
    (raise-user-error 'm1-cell "loader ~a: expected cifar-png or jpeg" name))
  (write-json (handoff-cell name workers #:warmup (setting 'warmup) #:epochs (setting 'epochs)
                            #:samples (setting 'samples) #:dataset (setting 'dataset)
                            #:trip (setting 'trip)))
  (newline))
