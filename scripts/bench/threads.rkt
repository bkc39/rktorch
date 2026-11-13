#lang racket/base

(require (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in torch + @ backward! randn relu sum tensor)
         (only-in racket/match match match-define)
         (only-in torch/nn Linear Sequential cross-entropy parameters sgd step! zero-grads!)
         (only-in "harness.rkt" bench-case clock-ms))

(provide thread-counts
         main-states
         mix-names
         ops-per-iteration
         scaling-run
         scaling-cases)

(define thread-counts '(1 2 4 8))
(define main-states '(idle train))
(define ops-per-iteration 4)
(define task-dir "/proc/self/task")
(define sample-interval 0.1)

(define (op-mix)
  (define a8 (randn 8 8))
  (define b8 (randn 8 8))
  (define a64 (randn 64 64))
  (define b64 (randn 64 64))
  (lambda ()
    (+ a8 b8)
    (@ a64 b64)
    (relu a64)
    (sum b64)))

(define (grain-mix)
  (define a (randn 256 256))
  (define b (randn 256 256))
  (lambda ()
    (@ a b)
    (+ a b)))

(define mixes
  (list (list 'small op-mix ops-per-iteration)
        (list 'grain grain-mix 2)))

(define mix-names (map car mixes))

(define (os-thread-count [tasks task-dir])
  (and (directory-exists? tasks) (length (directory-list tasks))))

(define (peak-count a b)
  (and a b (max a b)))

(define (start-peak-sampler #:tasks [tasks task-dir] #:interval [interval sample-interval])
  (define peak (box (os-thread-count tasks)))
  (define (sample!) (set-box! peak (peak-count (unbox peak) (os-thread-count tasks))))
  (define sampler
    (thread (lambda ()
              (let loop ()
                (sleep interval)
                (sample!)
                (loop)))))
  (lambda ()
    (kill-thread sampler)
    (sample!)
    (unbox peak)))

(define (training-step)
  (define net (Sequential (Linear 784 256) relu (Linear 256 10)))
  (define opt (sgd (parameters net) #:lr 0.01))
  (define xs (randn 64 784))
  (define ys (tensor (for/list ([i (in-range 64)]) (modulo i 10)) #:dtype 'int64))
  (lambda ()
    (zero-grads! opt)
    (backward! (cross-entropy (net xs) ys))
    (step! opt)))

(define (run-main state deadline)
  (match state
    ['idle (sync (alarm-evt deadline #t)) 0]
    ['train
     (define step! (training-step))
     (let loop ([steps 0])
       (cond
         [(>= (clock-ms) deadline) steps]
         [else (step!) (loop (add1 steps))]))]))

(define (start-workers n go? stop? counts failure make-mix)
  (define mixes (for/list ([_ (in-range n)]) (make-mix)))
  (define custodian (make-custodian-at-root))
  (define (fail! e)
    (let retry () (unless (or (unbox failure) (box-cas! failure #f e)) (retry)))
    (set-box! stop? #t))
  (values custodian
          (parameterize ([current-custodian custodian])
            (for/list ([mix (in-list mixes)] [i (in-naturals)])
              (thread (lambda ()
                        (with-handlers ([(lambda (_e) #t) fail!])
                          (let loop ([k 0])
                            (cond
                              [(not (unbox go?)) (loop k)]
                              [(and (positive? k) (unbox stop?)) (void)]
                              [else
                               (mix)
                               (vector-set! counts i (add1 k))
                               (loop (add1 k))]))))
                      #:pool 'own)))))

(define (reap! custodian workers grace)
  (for/sum ([w (in-list workers)])
    (cond
      [(sync/timeout grace w) 1]
      [else (parameterize ([current-custodian custodian]) (kill-thread w)) 0])))

(define (scaling-run n state
                     #:seconds seconds
                     #:grace [grace 30]
                     #:mix [make-mix op-mix]
                     #:ops-per-mix [ops-per-mix ops-per-iteration]
                     #:probe [probe void]
                     #:tasks [tasks task-dir])
  (define go? (box #f))
  (define stop? (box #f))
  (define counts (make-vector n 0))
  (define failure (box #f))
  (define-values (custodian workers) (start-workers n go? stop? counts failure make-mix))
  (define stop-sampler (start-peak-sampler #:tasks tasks))
  (define stopped #f)
  (define peak #f)
  (define probe-start (probe))
  (define t0 (clock-ms))
  (set-box! go? #t)
  (define-values (main-steps elapsed-s ops probe-stop)
    (dynamic-wind
     void
     (lambda ()
       (define steps (run-main state (+ t0 (* 1000.0 seconds))))
       (define elapsed (/ (- (clock-ms) t0) 1000.0))
       (define ops (for/sum ([k (in-vector counts)]) k))
       (values steps elapsed ops (probe)))
     (lambda ()
       (set! peak (stop-sampler))
       (set-box! stop? #t)
       (set! stopped (reap! custodian workers grace)))))
  (match (unbox failure)
    [#f (hasheq 'ops_per_s (/ (* ops-per-mix ops) elapsed-s)
                'mixes ops
                'elapsed_s elapsed-s
                'main_steps_per_s (/ main-steps elapsed-s)
                'started_ms t0
                'stopped stopped
                'peak_os_threads peak
                'probe_start probe-start
                'probe_stop probe-stop)]
    [e (raise e)]))

(define (scaling-cases #:seconds seconds
                       #:threads [threads thread-counts]
                       #:states [states main-states]
                       #:mix [mix 'small]
                       #:on-run [on-run void])
  (match-define (list _ make-mix ops-per-mix) (assq mix mixes))
  (for*/list ([state (in-list states)] [n (in-list threads)])
    (bench-case (string->symbol (format "~a/~a" state n))
                (lambda (_reps)
                  (define run (scaling-run n state #:seconds seconds #:mix make-mix
                                           #:ops-per-mix ops-per-mix))
                  (on-run state n run)
                  (hash-ref run 'ops_per_s))
                #:self-timed? #t)))

(module+ test
  (require (only-in racket/file make-temporary-directory)
           (only-in racket/match exn:misc:match?)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "harness.rkt" measure result-name result-per-call with-temporary-path))

  (define (touch! dir names)
    (for ([name (in-list names)])
      (call-with-output-file (build-path dir name) void)))

  (define absent "/nonexistent/task")

  (test-case "the OS thread count is the task directory's entries, and #f without one"
    (with-temporary-path (dir (make-temporary-directory "tasks-~a"))
      (touch! dir '("1" "2" "3"))
      (check-equal? (os-thread-count dir) 3))
    (check-false (os-thread-count absent))
    (if (directory-exists? "/proc/self/task")
        (check-true (exact-positive-integer? (os-thread-count)))
        (check-false (os-thread-count))))

  (test-case "the sampler keeps the most entries it saw, and #f without a directory"
    (with-temporary-path (dir (make-temporary-directory "tasks-~a"))
      (touch! dir '("1"))
      (define stop! (start-peak-sampler #:tasks dir #:interval 0.01))
      (touch! dir '("2" "3" "4"))
      (sleep 0.2)
      (for ([name (in-list '("2" "3" "4"))]) (delete-file (build-path dir name)))
      (sleep 0.2)
      (check-equal? (stop!) 4))
    (check-false ((start-peak-sampler #:tasks absent #:interval 0.01)))
    (check-false (peak-count 3 #f))
    (check-equal? (peak-count 3 5) 5))

  (test-case "a run records its peak OS threads: a thread per worker beside main, #f without /proc"
    (call-with-process-deadline
     120
     (lambda ()
       (define run (scaling-run 2 'idle #:seconds 0.3))
       (if (directory-exists? "/proc/self/task")
           (check-true (> (hash-ref run 'peak_os_threads) 2) (format "~a" run))
           (check-false (hash-ref run 'peak_os_threads)))
       (check-false (hash-ref (scaling-run 1 'idle #:seconds 0.1 #:tasks absent)
                              'peak_os_threads)))))

  (test-case "a mix counts its own ops: four for the small mix, two for the grain-crossing one"
    (call-with-process-deadline
     120
     (lambda ()
       (for ([mix (in-list mix-names)] [ops (in-list '(4 2))])
         (define runs '())
         (measure (scaling-cases #:seconds 0.1 #:threads '(1) #:states '(idle) #:mix mix
                                 #:on-run (lambda (_state _n run) (set! runs (cons run runs))))
                  #:warmup 0 #:rounds 1 #:load void)
         (match-define (list run) runs)
         (check-true (positive? (hash-ref run 'mixes)) (format "~a" mix))
         (check-= (/ (* (hash-ref run 'ops_per_s) (hash-ref run 'elapsed_s)) (hash-ref run 'mixes))
                  ops 1e-9 (format "~a" mix)))
       (check-exn exn:fail? (lambda () (scaling-cases #:seconds 0.1 #:mix 'huge))))))

  (test-case "workers on parallel threads count their ops, with main idle or training"
    (call-with-process-deadline
     240
     (lambda ()
       (define runs '())
       (define results
         (measure (scaling-cases #:seconds 0.2 #:threads '(1 2)
                                 #:on-run (lambda (state n run)
                                            (set! runs (cons (list state n run) runs))))
                  #:warmup 0 #:rounds 1 #:load void))
       (check-equal? (map result-name results) '(idle/1 idle/2 train/1 train/2))
       (for ([r (in-list results)] #:when (memq (result-name r) '(idle/1 idle/2)))
         (match-define (list ops) (result-per-call r))
         (check-true (positive? ops) (format "~a" (result-name r))))
       (for ([entry (in-list runs)])
         (match-define (list state n run) entry)
         (check-equal? (hash-ref run 'stopped) n)
         (check-equal? (zero? (hash-ref run 'main_steps_per_s)) (eq? state 'idle)))
       (check-equal? (length runs) 4))))

  (define (stalling)
    (define calls (box 0))
    (lambda ()
      (set-box! calls (add1 (unbox calls)))
      (define until (+ (clock-ms) (if (= (unbox calls) 1) 0.0 10000.0)))
      (let loop () (when (< (clock-ms) until) (loop)))))

  (test-case "a worker killed after the grace period keeps the ops it counted"
    (call-with-process-deadline
     120
     (lambda ()
       (define run (scaling-run 1 'idle #:seconds 0.3 #:grace 0.1 #:mix stalling))
       (check-true (positive? (hash-ref run 'ops_per_s)))
       (check-equal? (hash-ref run 'stopped) 0))))

  (test-case "no worker runs a mix before the clock starts"
    (call-with-process-deadline
     120
     (lambda ()
       (define stamps '())
       (define (stamping)
         (define first-call (box #f))
         (set! stamps (cons first-call stamps))
         (lambda () (unless (unbox first-call) (set-box! first-call (clock-ms)))))
       (for ([seconds (in-list '(0 0.1))])
         (set! stamps '())
         (define run (scaling-run 4 'idle #:seconds seconds #:mix stamping))
         (check-equal? (length stamps) 4)
         (for ([b (in-list stamps)])
           (check-pred real? (unbox b) "every worker ran its mix once")
           (check-true (>= (unbox b) (hash-ref run 'started_ms))))))))

  (test-case "a worker's raise stops the others and fails the run"
    (call-with-process-deadline
     120
     (lambda ()
       (define ticks (box 0))
       (define made (box 0))
       (define (tick!)
         (let loop () (unless (box-cas! ticks (unbox ticks) (add1 (unbox ticks))) (loop))))
       (define (failing-first)
         (set-box! made (add1 (unbox made)))
         (if (= (unbox made) 1) (lambda () (raise-user-error 'mix "unsupported")) tick!))
       (for ([seconds (in-list '(0 0.2))])
         (set-box! made 0)
         (check-exn #rx"mix: unsupported"
                    (lambda () (scaling-run 2 'idle #:seconds seconds #:mix failing-first)))
         (define seen (unbox ticks))
         (sync (alarm-evt (+ (clock-ms) 300) #t))
         (check-equal? (unbox ticks) seen "no worker ran after the raise")))))

  (test-case "workers stop when the main loop raises"
    (call-with-process-deadline
     120
     (lambda ()
       (define ticks (box 0))
       (define (ticking)
         (lambda () (let loop () (unless (box-cas! ticks (unbox ticks) (add1 (unbox ticks))) (loop)))))
       (check-exn exn:misc:match? (lambda () (scaling-run 2 'bogus #:seconds 0.1 #:mix ticking)))
       (define seen (unbox ticks))
       (check-true (>= seen 2) "each worker ran its mix once")
       (sync (alarm-evt (+ (clock-ms) 300) #t))
       (check-equal? (unbox ticks) seen "no worker ran after the raise")))))
