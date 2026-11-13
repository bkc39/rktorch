#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/string string-join)
         (only-in threading ~>)
         (only-in "markdown.rkt"
                  host iqr make-section median median-at median-of num percent ratio ref table whole))

(provide r-sections)

(define min-rounds 3)

(define row-keys '(workload main trip_mb k ffi between runner))

(define (row-key r)
  (for/hasheq ([key (in-list row-keys)]) (values key (ref r key))))

(define (key-without field r)
  (hash-remove (row-key r) field))

(define (sibling records r overrides)
  (define want
    (for/fold ([h (row-key r)]) ([(key value) (in-hash overrides)])
      (hash-set h key value)))
  (for/last ([x (in-list records)] #:when (equal? (row-key x) want)) x))

(define (lowest-k records r)
  (define key (key-without 'k r))
  (apply min (for/list ([x (in-list records)] #:when (equal? (key-without 'k x) key))
               (ref x 'k))))

(define (slowdown records r #:workload [workload (ref r 'workload)])
  (define at (sibling records r (hasheq 'workload workload)))
  (define base (and at (sibling records at (hasheq 'k (lowest-k records at)))))
  (ratio (median-of at) (median-of base)))

(define (spread r)
  (define m (median-of r))
  (define q (iqr r))
  (define n (ref r 'stats 'n))
  (and (real? m) (real? q) (positive? m) (real? n) (>= n min-rounds) (* 100.0 (/ q m))))

(define (spread-text r)
  (percent (spread r)))

(define (load-note r)
  (define limit (ref r 'max_load))
  (define loads (filter real? (append (or (ref r 'load_start) '()) (or (ref r 'load_end) '()))))
  (define peak (and (pair? loads) (apply max loads)))
  (if (and (real? limit) peak (>= peak limit)) (format "load ~a ≥ ~a" (num peak 1) limit) "-"))

(define (at r key [digits 2]) (num (median-at r key) digits))

(define (runtime-context r)
  (format "~a; OMP_NUM_THREADS=~a; ~a of ~a CPUs; run ~a" (host r)
          (or (ref r 'meta 'omp_num_threads) "unset") (or (ref r 'meta 'affinity_cpus) "?")
          (or (ref r 'meta 'cores) "?") (ref r 'run_id)))

(define (scaling-section records)
  (table '("workload" "main" "trip MB" "k" "wall ms" "spread" "÷ k=1" "CPU floor ÷ k=1"
           "GC ms" "main M allocs/s" "notes")
         (for/list ([r (in-list records)])
           (list (ref r 'workload) (ref r 'main) (ref r 'trip_mb) (ref r 'k)
                 (whole (median r)) (spread-text r) (slowdown records r)
                 (slowdown records r #:workload "cpu") (whole (median-at r 'work_gc_ms))
                 (at r 'main_rate 1) (load-note r)))))

(define (native-section records)
  (table '("worker call" "between calls" "call" "k" "main M allocs/s" "spread" "÷ none"
           "pause p50 ms" "pause p99 ms" "pause max ms" "worker calls/s" "GC ms" "trip MB"
           "notes")
         (for/list ([r (in-list records)])
           (define none (sibling records r (hasheq 'workload "none" 'ffi "none" 'between "none" 'k 0)))
           (list (ref r 'workload) (ref r 'between) (ref r 'ffi) (ref r 'k) (num (median r) 2)
                 (spread-text r) (ratio (median-of r) (median-of none))
                 (at r 'pause_p50_ms) (at r 'pause_p99_ms) (at r 'pause_max_ms)
                 (whole (median-at r 'worker_ops_per_s)) (whole (median-at r 'work_gc_ms))
                 (ref r 'trip_mb) (load-note r)))))

(define (places-section records)
  (define rows
    (remove-duplicates (map (lambda (r) (key-without 'runner r)) records)))
  (table '("workload" "trip MB" "k" "threads ms" "threads ÷ k=1" "places ms" "places ÷ k=1"
           "threads GC ms" "places main-place GC ms" "notes")
         (for/list ([key (in-list rows)])
           (define (runner name)
             (for/last ([r (in-list records)]
                        #:when (and (equal? (key-without 'runner r) key)
                                    (equal? (ref r 'runner) name)))
               r))
           (define threads (runner "threads"))
           (define places (runner "places"))
           (define (slow r) (if r (slowdown records r) "-"))
           (list (hash-ref key 'workload) (hash-ref key 'trip_mb) (hash-ref key 'k)
                 (whole (median-of threads)) (slow threads) (whole (median-of places)) (slow places)
                 (whole (median-at threads 'work_gc_ms)) (whole (median-at places 'work_gc_ms))
                 (~> (list threads places) (filter values _) (map load-note _) remove-duplicates
                     (string-join " "))))))

(define (ops-section records)
  (table '("op" "main" "call" "k" "main rate" "unit" "spread" "÷ no workers" "pause p50 ms"
           "pause p99 ms" "pause max ms" "worker ops/s" "µs/op/worker" "GC ms" "trip MB" "notes")
         (for/list ([r (in-list records)])
           (define none (sibling records r (hasheq 'workload "none" 'ffi "none" 'k 0)))
           (list (ref r 'workload) (ref r 'main) (ref r 'ffi) (ref r 'k) (num (median r) 2)
                 (ref r 'unit) (spread-text r) (ratio (median-of r) (median-of none))
                 (at r 'pause_p50_ms) (at r 'pause_p99_ms) (at r 'pause_max_ms)
                 (whole (median-at r 'worker_ops_per_s)) (~> r (median-at 'us_per_op) (num 1))
                 (whole (median-at r 'work_gc_ms)) (ref r 'trip_mb) (load-note r)))))

(define r-sections
  (list (make-section "runtime" "scaling" "R1. Pure-Racket scaling" scaling-section runtime-context)
        (make-section "runtime" "native" "R2. Native calls beside an allocating main"
                      native-section runtime-context)
        (make-section "runtime" "places" "R3. Places against threads" places-section
                      runtime-context)
        (make-section "runtime" "ops" "R4. Real ops in workers: plain, poll, collect-safe"
                      ops-section runtime-context)))

(module+ test
  (require rackunit
           (only-in "markdown.rkt" section-render))

  (define meta (hasheq 'host "lab" 'cpu "Some CPU" 'os "linux-x86_64" 'affinity "0-7"
                       'affinity_cpus 8 'cores 16 'omp_num_threads "1"))

  (define (record group workload k median
                  #:main [main "idle"] #:ffi [ffi "none"] #:between [between "none"]
                  #:runner [runner "threads"] #:n [n 1] #:iqr [iqr 0.0] #:unit [unit "ms"]
                  #:loads [loads '(1.0)] #:extra [extra (hasheq)])
    (for/fold ([h (hasheq 'suite "runtime" 'group group 'workload workload 'k k 'main main
                          'trip_mb "default" 'ffi ffi 'between between 'runner runner
                          'unit unit 'run_id "r1" 'meta meta 'max_load 4
                          'load_start loads 'load_end loads
                          'stats (hasheq 'median median 'iqr iqr 'n n))])
              ([(key value) (in-hash extra)])
      (hash-set h key value)))

  (define (render n records)
    ((section-render (list-ref r-sections n)) records))

  (define (row-of md . cells)
    (define pattern (~> cells (string-join " \\| ") (string-append "\\| " _ " \\|") regexp))
    (regexp-match? pattern md))

  (test-case "R1 puts each workload's slowdown beside the CPU floor's at the same k"
    (define md
      (render 0 (list (record "scaling" "cpu" 1 100.0)
                      (record "scaling" "cpu" 7 200.0)
                      (record "scaling" "alloc-retain" 1 100.0 #:n 3 #:iqr 10.0)
                      (record "scaling" "alloc-retain" 7 800.0 #:loads '(1.0 6.5)
                              #:extra (hasheq 'work_gc_ms '(900 1100) 'main_rate '(#f))))))
    (check-true (row-of md "cpu" "idle" "default" "1" "100" "-" "1.00" "1.00" "-" "-" "-") md)
    (check-true (row-of md "cpu" "idle" "default" "7" "200" "-" "2.00" "2.00") md)
    (check-true (row-of md "alloc-retain" "idle" "default" "1" "100" "10.0%" "1.00" "1.00") md)
    (check-true (row-of md "alloc-retain" "idle" "default" "7" "800" "-" "8.00" "2.00" "1000" "-"
                        "load 6.5 ≥ 4")
                md))

  (test-case "R2 sets each arm's main rate against the arm with no worker"
    (define (pauses p99) (hasheq 'pause_p50_ms '(0.1) 'pause_p99_ms (list p99)
                                 'pause_max_ms '(500.0) 'worker_ops_per_s '(400.0)))
    (define md
      (render 1 (list (record "native" "none" 0 50.0 #:main "alloc" #:unit "M allocs/s")
                      (record "native" "usleep-2000" 1 0.5 #:main "alloc" #:ffi "plain"
                              #:unit "M allocs/s" #:extra (pauses 300.0))
                      (record "native" "usleep-2000" 1 60.0 #:main "alloc" #:ffi "collect-safe"
                              #:unit "M allocs/s" #:extra (pauses 0.2)))))
    (check-true (row-of md "none" "none" "none" "0" "50.00" "-" "1.00") md)
    (check-true (row-of md "usleep-2000" "none" "plain" "1" "0.50" "-" "0.01" "0.10" "300.00"
                        "500.00" "400")
                md)
    (check-true (row-of md "usleep-2000" "none" "collect-safe" "1" "60.00" "-" "1.20" "0.10" "0.20")
                md))

  (test-case "R3 sets places beside threads, each against its own k=1"
    (define md
      (render 2 (list (record "places" "alloc-retain" 1 100.0)
                      (record "places" "alloc-retain" 7 1000.0)
                      (record "places" "alloc-retain" 1 120.0 #:runner "places")
                      (record "places" "alloc-retain" 7 960.0 #:runner "places"))))
    (check-true (row-of md "alloc-retain" "default" "1" "100" "1.00" "120" "1.00") md)
    (check-true (row-of md "alloc-retain" "default" "7" "1000" "10.00" "960" "8.00") md)
    (check-equal? (length (regexp-match* #rx"\n\\| alloc-retain" md)) 2))

  (test-case "R4 sets each op arm's main rate against main with no workers"
    (define md
      (render 3 (list (record "ops" "none" 0 400.0 #:main "train" #:unit "steps/s")
                      (record "ops" "mm1024" 7 20.0 #:main "train" #:ffi "plain" #:unit "steps/s"
                              #:extra (hasheq 'us_per_op '(25000.0)))
                      (record "ops" "mm1024" 7 160.0 #:main "train" #:ffi "collect-safe"
                              #:unit "steps/s"))))
    (check-true (row-of md "mm1024" "train" "plain" "7" "20.00" "steps/s" "-" "0.05" "-" "-" "-"
                        "-" "25000.0")
                md)
    (check-true (row-of md "mm1024" "train" "collect-safe" "7" "160.00" "steps/s" "-" "0.40") md))

  (test-case "Table R is headed by the host, its OMP setting and the CPUs it may use"
    (check-equal? (runtime-context (record "scaling" "cpu" 1 1.0))
                  "lab (Some CPU, linux-x86_64, cpus 0-7); OMP_NUM_THREADS=1; 8 of 16 CPUs; run r1")
    (check-regexp-match #rx"OMP_NUM_THREADS=unset; \\? of \\? CPUs"
                        (runtime-context (hash-set (record "scaling" "cpu" 1 1.0) 'meta (hasheq))))))
