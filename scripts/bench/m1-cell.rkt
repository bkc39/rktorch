#lang racket/base

(require (only-in racket/dict in-dict)
         (only-in racket/list last remove-duplicates)
         (only-in racket/match match match* match-define)
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
         (only-in "harness.rkt" mean)
         (only-in "m1-clocks.rkt"
                  gc-event-cpu-ms gc-event-end gc-event-live gc-event-mode gc-event-start
                  gc-overlap-ms os-thread-id published-entries span span-kind span-tid span-wall₀
                  span-wall₁ spans task-cpu-ms thread-cpu-ms timed with-gc-log)
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

(struct idle (head tail gaps absent))

(define no-idle (idle 0.0 0.0 0.0 0.0))

(define (idle+ a b)
  (match* (a b)
    [((idle h t g x) (idle h* t* g* x*)) (idle (+ h h*) (+ t t*) (+ g g*) (+ x x*))]))

(define (idle->hash i)
  (match-define (idle h t g x) i)
  (hasheq 'head_ms h 'tail_ms t 'gaps_ms g 'absent_ms x))

(define (span-wall s) (- (span-wall₁ s) (span-wall₀ s)))

(define (span-cpu s)
  (match s
    [(span _ _ _ _ (? real? c₀) (? real? c₁)) (- c₁ c₀)]
    [_ 0.0]))

(define (clipped s e₀ e₁)
  (define a (max e₀ (span-wall₀ s)))
  (define b (min e₁ (span-wall₁ s)))
  (and (< a b) (cons a b)))

(define (one-worker-idle intervals e₀ e₁)
  (define sorted (sort intervals < #:key car))
  (match-define (cons (cons start _) _) sorted)
  (match-define (cons _ end) (last sorted))
  (idle (- start e₀)
        (- e₁ end)
        (for/sum ([a (in-list sorted)] [b (in-list (cdr sorted))]) (- (car b) (cdr a)))
        0.0))

(define (epoch-idle e builds workers)
  (define e₀ (span-wall₀ e))
  (define e₁ (span-wall₁ e))
  (define inside
    (for*/list ([b (in-list builds)] [c (in-value (clipped b e₀ e₁))] #:when c)
      (cons (span-tid b) c)))
  (define tids (remove-duplicates (map car inside)))
  (for/fold ([acc (idle 0.0 0.0 0.0 (* (max 0 (- workers (length tids))) (- e₁ e₀)))])
            ([t (in-list tids)])
    (idle+ acc (one-worker-idle (for/list ([(tid interval) (in-dict inside)] #:when (eqv? tid t))
                                  interval)
                                e₀ e₁))))

(define (worker-idle epochs builds workers)
  (for/fold ([acc no-idle]) ([e (in-list epochs)] #:when (positive? workers))
    (idle+ acc (epoch-idle e builds workers))))

(define (outside-epochs-ms builds epochs)
  (for/sum ([b (in-list builds)])
    (- (span-wall b)
       (for*/sum ([e (in-list epochs)] [c (in-value (clipped b (span-wall₀ e) (span-wall₁ e)))]
                  #:when c)
         (- (cdr c) (car c))))))

(define (totals ss events)
  (define wall (for/sum ([s (in-list ss)]) (span-wall s)))
  (define cpu (for/sum ([s (in-list ss)]) (span-cpu s)))
  (hasheq 'wall_ms wall
          'cpu_ms cpu
          'off_ms (- wall cpu)
          'gc_ms (for/sum ([s (in-list ss)]) (gc-overlap-ms events (span-wall₀ s) (span-wall₁ s)))
          'count (length ss)))

(define (spans-of all kind [where (lambda (_s) #t)])
  (filter (lambda (s) (and (eq? (span-kind s) kind) (where s))) all))

(define (span-split all events #:main main #:workers [workers 0])
  (define (on-main s) (eqv? (span-tid s) main))
  (define (off-main s) (not (on-main s)))
  (define (of kind [where (lambda (_s) #t)]) (totals (spans-of all kind where) events))
  (define epochs (spans-of all 'epoch))
  (define builds (spans-of all 'build off-main))
  (hasheq 'epoch (of 'epoch)
          'start (of 'start)
          'finish (of 'finish)
          'next (of 'next)
          'step (of 'step)
          'main_build (of 'build on-main)
          'main_collate (of 'collate on-main)
          'worker_build (hash-set (totals builds events)
                                  'outside_ms (outside-epochs-ms builds epochs))
          'worker_collate (of 'collate off-main)
          'worker_threads (~> (map span-tid builds) remove-duplicates length)
          'worker_idle (idle->hash (worker-idle epochs builds workers))))

(define (gc-real e) (- (gc-event-end e) (gc-event-start e)))

(define (gc-row es)
  (hasheq 'count (length es)
          'real_ms (for/sum ([e (in-list es)]) (gc-real e))
          'cpu_ms (for/sum ([e (in-list es)]) (gc-event-cpu-ms e))
          'mean_real_ms (mean (map gc-real es))
          'mean_live (mean (filter real? (map gc-event-live es)))))

(define (gc-summary events seconds)
  (define (mode-of m) (filter (lambda (e) (eq? (gc-event-mode e) m)) events))
  (hasheq 'all (gc-row events)
          'minor (gc-row (mode-of 'minor))
          'major (gc-row (mode-of 'major))
          'incremental (gc-row (mode-of 'incremental))
          'per_s (/ (length events) seconds)
          'points (for/list ([e (in-list events)])
                    (list (symbol->string (gc-event-mode e)) (gc-real e) (gc-event-live e)))))

(define (cpu-by-task before after main workers-cpu)
  (define (moved tid) (- (hash-ref after tid 0.0) (hash-ref before tid 0.0)))
  (define persistent (filter (lambda (t) (hash-has-key? after t)) (hash-keys before)))
  (hasheq 'main_ms (moved main)
          'other_persistent_ms (for/sum ([t (in-list persistent)] #:unless (= t main)) (moved t))
          'workers_span_ms workers-cpu))

(define (handoff-cell name workers #:warmup warmup #:epochs epochs #:samples samples
                      #:dataset [dataset "full"] #:trip [trip #f] #:count? [count? (zero? workers)])
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
      (with-gc-log (#:live published-entries)
        (for/sum ([_ (in-range epochs)]) (epoch! loader))))
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
           (only-in racket/cmdline command-line)
           (only-in "flags.rkt" flag-count flag-positive))

  (define (cell-settings argv)
    (define settings (hasheq 'warmup 1 'epochs 2 'samples #f 'dataset "full" 'trip #f))
    (define (put! key value) (set! settings (hash-set settings key value)))
    (define (count text [ok? exact-positive-integer?]) (flag-count 'm1-cell text ok?))
    (define-values (name workers)
      (command-line
       #:program "m1-cell"
       #:argv argv
       #:once-each
       [("--warmup") n "Untimed epochs" (put! 'warmup (count n exact-nonnegative-integer?))]
       [("--epochs") n "Timed epochs" (put! 'epochs (count n))]
       [("--samples") n "Samples per epoch" (put! 'samples (count n))]
       [("--dataset") d "The CIFAR-10 PNG tree: full or fixture" (put! 'dataset d)]
       [("--trip") mb "collect-trip-bytes in MB while the cell runs (default: Racket's)"
                   (put! 'trip (flag-positive 'm1-cell mb))]
       #:args (_loader name workers)
       (values (string->symbol name) (count workers exact-nonnegative-integer?))))
    (unless (memq name cell-pipelines)
      (raise-user-error 'm1-cell "loader ~a: expected cifar-png or jpeg" name))
    (hash-set* settings 'name name 'workers workers))

  (define settings (cell-settings (current-command-line-arguments)))
  (define (setting key) (hash-ref settings key))
  (write-json (handoff-cell (setting 'name) (setting 'workers)
                            #:warmup (setting 'warmup) #:epochs (setting 'epochs)
                            #:samples (setting 'samples) #:dataset (setting 'dataset)
                            #:trip (setting 'trip)))
  (newline))
