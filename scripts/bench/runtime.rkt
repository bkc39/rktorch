#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/format ~a ~r)
         (only-in racket/list append* take-right)
         (only-in racket/future processor-count)
         (only-in racket/match match match*)
         (only-in racket/place dynamic-place place-channel-put place-dead-evt)
         (only-in racket/runtime-path define-runtime-path)
         (only-in racket/string string-split)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in threading ~>)
         (only-in "harness.rkt"
                  bench-case clock-ms load-average measure parse-cpu-times quantile result->record
                  result-per-call)
         (only-in "runtime-work.rkt"
                  allocate-until native-call native-calls run-work with-trip-bytes work-units
                  workload-names)
         (only-in "threads.rkt" call-with-workers))

(provide allocate-until-deadline
         default-workers
         cell
         cell-main
         cell-workload
         cell-ffi
         cell-records
         duration-run
         scaling-cells
         native-cells
         places-cells
         run-cell)

(define-runtime-path work-module "runtime-work.rkt")

(define grace 30)

(define (default-workers cpus)
  (filter (lambda (k) (<= k (max 1 (sub1 cpus)))) '(1 4 7)))

(struct cell (group workload main k trip ffi between runner units seconds)
  #:constructor-name make-cell
  #:omit-define-syntaxes
  #:transparent)

(define (cell #:group group
              #:workload workload
              #:k k
              #:main [main 'idle]
              #:trip [trip #f]
              #:ffi [ffi 'none]
              #:between [between 'none]
              #:runner [runner 'threads]
              #:units [units #f]
              #:seconds [seconds #f])
  (make-cell group workload main k trip ffi between runner units seconds))

(define (cell-name c)
  (string->symbol
   (format "~a/~a/~a/~a/~a/~a/k~a/trip-~a" (cell-group c) (cell-runner c) (cell-workload c)
           (cell-ffi c) (cell-between c) (cell-main c) (cell-k c) (or (cell-trip c) "default"))))

(define (rate-scale c)
  (if (eq? (cell-main c) 'train) 1.0 1e6))

(define (cell-unit c)
  (match* ((cell-group c) (cell-main c))
    [((or 'scaling 'places) _) "ms"]
    [(_ 'train) "steps/s"]
    [(_ _) "M allocs/s"]))

(define (primary c)
  (if (memq (cell-group c) '(scaling places)) 'wall_ms 'main_rate))

(define metric-keys
  '(wall_ms work_gc_ms main_rate pause_p50_ms pause_p99_ms pause_max_ms worker_ops_per_s
            us_per_op stopped load_outside))

(define (pause-stats gaps)
  (if (null? gaps)
      (hasheq 'pause_p50_ms #f 'pause_p99_ms #f 'pause_max_ms #f)
      (hasheq 'pause_p50_ms (quantile gaps 0.5)
              'pause_p99_ms (quantile gaps 0.99)
              'pause_max_ms (apply max gaps))))

(define (main-stats c n gaps elapsed-ms gc-ms)
  (hash-set* (pause-stats gaps)
             'wall_ms elapsed-ms
             'work_gc_ms gc-ms
             'main_rate (and n (positive? elapsed-ms) (/ (* 1000.0 n) elapsed-ms (rate-scale c)))))

(define (call-with-shutdown thunk)
  (define custodian (make-custodian))
  (dynamic-wind void
                (lambda () (parameterize ([current-custodian custodian]) (thunk)))
                (lambda () (custodian-shutdown-all custodian))))

(define-syntax-parse-rule (with-shutdown body:expr ...+)
  (call-with-shutdown (lambda () body ...)))

(define (call-with-finishing-workers k work proc)
  (define go? (box #f))
  (define failure (box #f))
  (define (fail! e) (box-cas! failure #f e))
  (define result
    (with-shutdown
      (define workers
        (for/list ([_ (in-range k)])
          (thread (lambda ()
                    (with-handlers ([(lambda (_e) #t) fail!])
                      (let wait () (unless (unbox go?) (wait)))
                      (work)))
                  #:pool 'own)))
      (proc (lambda () (set-box! go? #t))
            (lambda () (andmap thread-dead? workers))
            (lambda () (for-each thread-wait workers)))))
  (cond
    [(unbox failure) => raise]
    [else result]))

(define (run-scaling c)
  (define name (cell-workload c))
  (define units (cell-units c))
  (call-with-finishing-workers
   (cell-k c) (lambda () (run-work name units))
   (lambda (go! done? join)
     (define gc0 (current-gc-milliseconds))
     (define t0 (clock-ms))
     (go!)
     (define-values (n gaps)
       (match (cell-main c)
         ['idle (join) (values #f '())]
         ['alloc (allocate-until (lambda (_now) (done?)) #:start t0)]))
     (join)
     (main-stats c n gaps (- (clock-ms) t0) (- (current-gc-milliseconds) gc0)))))

(define (expect-message p message)
  (define got (sync (handle-evt (place-dead-evt p) (lambda (_) 'dead)) p))
  (unless (equal? got message)
    (raise-user-error 'runtime "a place sent ~s, not ~s" got message)))

(define (run-places c #:collect [collect collect-garbage] #:expect [expect expect-message])
  (with-shutdown
    (define places
      (for/list ([_ (in-range (cell-k c))]) (dynamic-place work-module 'place-worker)))
    (for ([p (in-list places)]) (place-channel-put p (list (cell-workload c) (cell-units c))))
    (for ([p (in-list places)]) (expect p 'ready))
    (collect)
    (define gc0 (current-gc-milliseconds))
    (define t0 (clock-ms))
    (for ([p (in-list places)]) (place-channel-put p 'go))
    (for ([p (in-list places)]) (expect p 'done))
    (main-stats c #f '() (- (clock-ms) t0) (- (current-gc-milliseconds) gc0))))

(define (duration-run c make-call main-loop)
  (define k (cell-k c))
  (define-values (run stopped)
    (call-with-workers
     k make-call grace
     (lambda (go! ops-done)
       (define gc0 (current-gc-milliseconds))
       (define t0 (clock-ms))
       (go!)
       (define-values (n gaps) (~> c cell-seconds (* 1000.0) (+ t0) main-loop))
       (define elapsed (- (clock-ms) t0))
       (define ops (ops-done))
       (hash-set* (main-stats c n gaps elapsed (- (current-gc-milliseconds) gc0))
                  'worker_ops_per_s (/ (* 1000.0 ops) elapsed)
                  'us_per_op (and (positive? ops) (/ (* 1000.0 elapsed k) ops))))))
  (hash-set run 'stopped stopped))

(define (allocate-until-deadline deadline)
  (allocate-until (lambda (now) (>= now deadline))))

(define (run-native c)
  (duration-run c
                (lambda () (native-call (cell-workload c) (cell-ffi c) (cell-between c)))
                allocate-until-deadline))

(define (run-cell c)
  (match* ((cell-group c) (cell-runner c))
    [(_ 'places) (run-places c)]
    [('native _) (run-native c)]
    [(_ _) (run-scaling c)]))

(define (scaling-cells #:workers ks
                       #:trips trips
                       #:scale scale
                       #:workloads [names workload-names]
                       #:mains [mains '(idle alloc)])
  (for*/list ([trip (in-list trips)] [main (in-list mains)] [name (in-list names)] [k (in-list ks)])
    (cell #:group 'scaling #:workload name #:main main #:k k #:trip trip
          #:units (work-units name scale))))

(define (native-cells #:workers ks
                      #:trips trips
                      #:seconds seconds
                      #:calls [calls (map car native-calls)]
                      #:ffis [ffis '(plain poll collect-safe)]
                      #:betweens [betweens '(none alloc)])
  (append*
   (for/list ([trip (in-list trips)])
     (cons (cell #:group 'native #:workload 'none #:main 'alloc #:k 0 #:trip trip #:seconds seconds)
           (for*/list ([call (in-list calls)]
                       [between (in-list betweens)]
                       [ffi (in-list ffis)]
                       [k (in-list ks)])
             (cell #:group 'native #:workload call #:main 'alloc #:k k #:trip trip #:ffi ffi
                   #:between between #:seconds seconds))))))

(define (places-cells #:workers ks #:trips trips #:scale scale #:workload [name 'alloc-retain])
  (for*/list ([trip (in-list trips)] [runner (in-list '(threads places))] [k (in-list ks)])
    (cell #:group 'places #:workload name #:k k #:trip trip #:runner runner
          #:units (work-units name scale))))

(define (cell-extra c timed max-load)
  (for/fold ([h (hasheq 'impl "rktorch"
                        'case (~a (cell-workload c))
                        'variant (~a (cell-name c))
                        'workload (~a (cell-workload c))
                        'k (cell-k c)
                        'main (~a (cell-main c))
                        'trip_mb (or (cell-trip c) "default")
                        'ffi (~a (cell-ffi c))
                        'between (~a (cell-between c))
                        'runner (~a (cell-runner c))
                        'units (cell-units c)
                        'seconds (cell-seconds c)
                        'max_load max-load)])
            ([key (in-list metric-keys)])
    (hash-set h key (map (lambda (run) (hash-ref run key #f)) timed))))

(define (first-line path)
  (and (file-exists? path) (call-with-input-file path read-line)))

(define (parse-self-ticks text)
  (match (and (string? text) (regexp-match #px"^.*\\)\\s+(.*)$" text))
    [(list _ rest)
     (match (map string->number (string-split rest))
       [(list _ _ _ _ _ _ _ _ _ _ _ utime stime _ ...)
        #:when (and (exact-nonnegative-integer? utime) (exact-nonnegative-integer? stime))
        (+ utime stime)]
       [_ #f])]
    [_ #f]))

(define (process-cpu-times)
  (define system (parse-cpu-times (first-line "/proc/stat")))
  (define self (parse-self-ticks (first-line "/proc/self/stat")))
  (and system self (list (car system) (cdr system) self)))

(define (outside-cores before after cpus)
  (match* (before after)
    [((list busy₀ total₀ self₀) (list busy₁ total₁ self₁))
     #:when (> total₁ total₀)
     (max 0.0 (/ (* cpus (- (- busy₁ busy₀) (- self₁ self₀))) (- total₁ total₀) 1.0))]
    [(_ _) #f]))

(define (cell-records cells run
                      #:rounds rounds
                      #:warmup warmup
                      #:meta meta
                      #:max-load max-load
                      #:load [load load-average]
                      #:cpu-times [cpu-times process-cpu-times]
                      #:log [log (current-error-port)])
  (define runs (make-hash))
  (define (load-less k)
    (match (load)
      [(cons one-minute _) (max 0.0 (- one-minute k 1))]
      [_ #f]))
  (define (case-of c)
    (bench-case (cell-name c)
                (lambda (_reps)
                  (define before (cpu-times))
                  (define result (with-trip-bytes (cell-trip c) (run c)))
                  (define outside
                    (or (outside-cores before (cpu-times) (processor-count)) (load-less (cell-k c))))
                  (define value (or (hash-ref result (primary c)) 0.0))
                  (hash-update! runs c (lambda (rs) (cons (hash-set result 'load_outside outside) rs))
                                '())
                  (fprintf log "  ~a ~a ~a\n" (cell-name c) (~r value #:precision 2) (cell-unit c))
                  value)
                #:before collect-garbage
                #:self-timed? #t))
  (define results (measure (map case-of cells) #:warmup warmup #:rounds rounds #:load load))
  (for/list ([c (in-list cells)] [r (in-list results)])
    (define timed
      (~> (hash-ref runs c '()) reverse (take-right (length (result-per-call r)))))
    (result->record r #:suite "runtime" #:group (~a (cell-group c)) #:unit (cell-unit c)
                    #:meta meta #:scale 1 #:extra (cell-extra c timed max-load))))

(module+ test
  (require (only-in racket/file make-temporary-directory)
           (only-in racket/list first last second third)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "markdown.rkt" section-render)
           (only-in "summary-r.rkt" r-sections))

  (define quiet (open-output-nowhere))

  (define (records-of cells
                      #:rounds [rounds 1]
                      #:warmup [warmup 0]
                      #:run [run run-cell]
                      #:max-load [max-load 99]
                      #:load [load (lambda () '(1.0 2.0 3.0))]
                      #:cpu-times [cpu-times (lambda () #f)])
    (cell-records cells run #:rounds rounds #:warmup warmup #:meta (hasheq 'host "test")
                  #:max-load max-load #:load load #:cpu-times cpu-times #:log quiet))

  (define (fake-run _c)
    (hasheq 'wall_ms 1.0 'main_rate #f))

  (define (only r key) (car (hash-ref r key)))

  (define (row r)
    (map (lambda (key) (hash-ref r key)) '(group workload main k trip_mb ffi between runner)))

  (test-case "the default workers stop one short of the CPUs"
    (check-equal? (default-workers 16) '(1 4 7))
    (check-equal? (default-workers 8) '(1 4 7))
    (check-equal? (default-workers 5) '(1 4))
    (check-equal? (default-workers 2) '(1))
    (check-equal? (default-workers 1) '(1)))

  (test-case "every scaling workload runs, main idle or allocating, at each trip"
    (call-with-process-deadline
     120
     (lambda ()
       (define records
         (records-of (scaling-cells #:workers '(1 2) #:trips '(#f 16) #:scale 1e-4)
                     #:rounds 2))
       (check-equal? (length records) (* 2 2 (length workload-names) 2))
       (for ([r (in-list records)])
         (check-equal? (hash-ref r 'suite) "runtime")
         (check-equal? (hash-ref r 'unit) "ms")
         (check-equal? (hash-ref r 'meta) (hasheq 'host "test"))
         (check-equal? (hash-ref r 'max_load) 99)
         (check-equal? (hash-ref r 'load_outside) '(0.0 0.0))
         (check-equal? (length (hash-ref r 'wall_ms)) 2)
         (check-equal? (hash-ref r 'samples) (hash-ref r 'wall_ms))
         (check-true (andmap positive? (hash-ref r 'wall_ms)) (format "~a" (row r)))
         (check-equal? (real? (only r 'main_rate)) (equal? (hash-ref r 'main) "alloc")
                       (format "~a" (row r))))
       (check-equal? (row (car records)) '("scaling" "cpu" "idle" 1 "default" "none" "none" "threads"))
       (check-equal? (row (list-ref records 1)) '("scaling" "cpu" "idle" 2 "default" "none" "none"
                                                            "threads"))
       (check-equal? (hash-ref (last records) 'trip_mb) 16))))

  (test-case "a worker's raise fails a scaling run"
    (call-with-process-deadline
     60
     (lambda ()
       (for ([main (in-list '(idle alloc))])
         (check-exn exn:fail?
                    (lambda ()
                      (run-cell (cell #:group 'scaling #:workload 'nope #:main main #:k 2
                                      #:units 1))))))))

  (test-case "native arms time main's allocation beside workers in a foreign call"
    (call-with-process-deadline
     120
     (lambda ()
       (define records
         (records-of (native-cells #:workers '(1 2) #:trips '(#f) #:seconds 0.1
                                   #:calls '(usleep-200))))
       (check-equal? (length records) (add1 (* 2 3 2)))
       (define none (car records))
       (check-equal? (row none) '("native" "none" "alloc" 0 "default" "none" "none" "threads"))
       (check-true (zero? (only none 'worker_ops_per_s)))
       (for ([r (in-list records)])
         (check-equal? (hash-ref r 'unit) "M allocs/s")
         (check-true (positive? (only r 'main_rate)) (format "~a" (row r)))
         (check-equal? (only r 'stopped) (hash-ref r 'k))
         (for ([key (in-list '(pause_p50_ms pause_p99_ms pause_max_ms))])
           (check-pred real? (only r key) (format "~a ~a" key (row r))))
         (unless (zero? (hash-ref r 'k))
           (check-true (positive? (only r 'worker_ops_per_s)) (format "~a" (row r)))
           (check-pred real? (only r 'us_per_op)))))))

  (test-case "main's p99 pause beside a 2 ms collect-safe call is below the plain call's"
    (call-with-process-deadline
     120
     (lambda ()
       (define (p99 ffi)
         (define c (cell #:group 'native #:workload 'usleep-2000 #:main 'alloc #:k 1 #:ffi ffi
                         #:seconds 1.0))
         (hash-ref (run-cell c) 'pause_p99_ms))
       (define plain (p99 'plain))
       (define safe (p99 'collect-safe))
       (check-true (< safe plain) (format "collect-safe ~a ms, plain ~a ms" safe plain)))))

  (test-case "places and threads time the work phase, the place module found from any directory"
    (call-with-process-deadline
     180
     (lambda ()
       (with-temporary-path (dir (make-temporary-directory "runtime-~a"))
         (parameterize ([current-directory dir])
           (define records (records-of (places-cells #:workers '(1 2) #:trips '(#f) #:scale 1e-3)))
           (check-equal? (map (lambda (r) (list (hash-ref r 'runner) (hash-ref r 'k))) records)
                         '(("threads" 1) ("threads" 2) ("places" 1) ("places" 2)))
           (for ([r (in-list records)])
             (check-equal? (hash-ref r 'workload) "alloc-retain")
             (check-equal? (hash-ref r 'main) "idle")
             (check-true (positive? (only r 'wall_ms)) (format "~a" (row r))))))
       (check-exn #rx"a place sent \\(failed \"workload-run: [^\n]*, not done"
                  (lambda ()
                    (run-cell (cell #:group 'places #:runner 'places #:workload 'nope #:k 1
                                    #:units 1)))))))

  (test-case "the outside load is the busy cores less this process's own CPU time"
    (check-equal? (parse-self-ticks
                   "42 (a (b) c) R 1 2 3 0 -1 4194304 489 0 0 0 120 30 0 0 25 5 1 0 3 4 5")
                  150)
    (check-false (parse-self-ticks "42 (racket) R 1 2"))
    (check-false (parse-self-ticks #f))
    (check-equal? (outside-cores '(100 1000 40) '(400 2000 240) 8) 0.8)
    (check-equal? (outside-cores '(100 1000 40) '(400 2000 400) 8) 0.0)
    (check-false (outside-cores #f '(400 2000 240) 8))
    (check-false (outside-cores '(100 1000 40) '(400 1000 240) 8))
    (define now (process-cpu-times))
    (when now
      (check-true (andmap exact-nonnegative-integer? now))
      (check-true (<= (third now) (first now)))))

  (test-case "a load that rises by the suite's own k workers does not fire the load note"
    (define ticks (box '(1000 10000 50)))
    (define (busy-as-self)
      (define now (unbox ticks))
      (set-box! ticks (list (+ (first now) 500) (+ (second now) 1000) (+ (third now) 500)))
      now)
    (define cells (scaling-cells #:workers '(1 4 7) #:trips '(#f) #:scale 1 #:workloads '(cpu)
                                 #:mains '(idle)))
    (define (rendered records)
      ((section-render (car r-sections)) records))
    (define measured (records-of cells #:run fake-run #:max-load 4 #:cpu-times busy-as-self))
    (check-equal? (map (lambda (r) (hash-ref r 'load_outside)) measured) '((0.0) (0.0) (0.0)))
    (check-false (regexp-match? #rx"load [0-9]" (rendered measured)))
    (define fallback (records-of cells #:run fake-run #:max-load 4 #:load (lambda () '(8.5 1.0 1.0))))
    (check-equal? (map (lambda (r) (hash-ref r 'load_outside)) fallback) '((6.5) (3.5) (0.5)))
    (check-regexp-match #rx"load 6\\.5 ≥ 4" (rendered fallback))
    (check-false (regexp-match? #rx"load 3\\.5" (rendered fallback))))

  (test-case "places collect after every place is ready and before the timed phase"
    (call-with-process-deadline
     120
     (lambda ()
       (define events '())
       (define (note! e) (set! events (cons e events)))
       (define c (cell #:group 'places #:runner 'places #:workload 'cpu #:k 2 #:units 10))
       (run-places c
                   #:collect (lambda () (note! 'collect) (collect-garbage))
                   #:expect (lambda (p message) (expect-message p message) (note! message)))
       (check-equal? (reverse events) '(ready ready collect done done)))))

  (test-case "warmup runs are left out of a record's rounds"
    (call-with-process-deadline
     60
     (lambda ()
       (define cells (scaling-cells #:workers '(1) #:trips '(#f) #:scale 1e-5 #:workloads '(cpu)
                                    #:mains '(idle)))
       (define r (car (records-of cells #:rounds 2 #:warmup 1)))
       (check-equal? (length (hash-ref r 'wall_ms)) 2)
       (check-equal? (hash-ref (hash-ref r 'stats) 'n) 2)))))
