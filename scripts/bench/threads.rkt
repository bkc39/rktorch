#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in racket/match match match-define)
         (only-in racket/promise delay force)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in torch + @ backward! randn relu sum tensor)
         (only-in torch/nn Linear Sequential cross-entropy parameters sgd step! zero-grads!)
         (only-in "harness.rkt" bench-case clock-ms))

(provide call-with-workers
         thread-counts
         main-states
         mix-names
         ops-per-iteration
         peak-of
         scaling-run
         scaling-cases
         training-step)

(define thread-counts '(1 2 4 8))
(define main-states '(idle train))
(define ops-per-iteration 4)
(define task-dir "/proc/self/task")
(define sample-interval 0.1)

(define (small-mix)
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
  (define inputs (delay (list (randn 256 256) (randn 256 256))))
  (lambda ()
    (match-define (list a b) (force inputs))
    (@ a b)
    (+ a b)))

(struct op-mix (name make ops))

(define mixes
  (list (op-mix 'small small-mix ops-per-iteration)
        (op-mix 'grain grain-mix 2)))

(define mix-names (map op-mix-name mixes))

(define (mix-named name)
  (findf (lambda (m) (eq? (op-mix-name m) name)) mixes))

(define (os-thread-count [tasks task-dir])
  (and (directory-exists? tasks) (length (directory-list tasks))))

(define (settled-thread-count tasks #:interval [interval sample-interval] #:within [limit 2.0])
  (define deadline (+ (clock-ms) (* 1000.0 limit)))
  (let loop ([last (os-thread-count tasks)])
    (sleep interval)
    (define now (os-thread-count tasks))
    (if (or (equal? now last) (>= (clock-ms) deadline)) now (loop now))))

(define (peak-of counts)
  (and (andmap values counts) (apply max counts)))

;; Chez's collector starts a sweeper OS thread, up to 16, when more threads
;; wait on a collection than it has sweepers, and keeps it for the life of the
;; process; priming past a run's threads keeps the run from starting any.
(define sweeper-spinners 32)

(define (collect-with-allocating-threads! k)
  (define started (make-semaphore 0))
  (define stop? (box #f))
  (define spinners
    (for/list ([_ (in-range k)])
      (thread (lambda ()
                (define sink (box #f))
                (semaphore-post started)
                (let loop ()
                  (unless (unbox stop?)
                    (set-box! sink (make-vector 16))
                    (loop))))
              #:pool 'own)))
  (dynamic-wind void
                (lambda ()
                  (for ([_ (in-range k)]) (semaphore-wait started))
                  (collect-garbage 'major))
                (lambda ()
                  (set-box! stop? #t)
                  (for-each thread-wait spinners))))

(define (start-collector-sweepers! tasks)
  (let loop ([before (settled-thread-count tasks)] [rounds 4])
    (collect-with-allocating-threads! sweeper-spinners)
    (define after (settled-thread-count tasks))
    (if (or (equal? after before) (zero? rounds)) after (loop after (sub1 rounds)))))

(define collector-sweepers (delay (start-collector-sweepers! task-dir)))

(define (call-with-peak-sampler proc #:tasks [tasks task-dir] #:interval [interval sample-interval])
  (define peak (box (os-thread-count tasks)))
  (define (sample!) (set-box! peak (peak-of (list (unbox peak) (os-thread-count tasks)))))
  (define sampler
    (thread (lambda ()
              (let loop ()
                (sleep interval)
                (sample!)
                (loop)))))
  (define result (dynamic-wind void proc (lambda () (kill-thread sampler))))
  (sample!)
  (values result (unbox peak)))

(define-syntax-parse-rule (with-peak-sampler ((~alt (~optional (~seq #:tasks tasks:expr))
                                                    (~optional (~seq #:interval interval:expr)))
                                              ...)
                            body:expr ...+)
  (call-with-peak-sampler (lambda () body ...)
                          (~? (~@ #:tasks tasks))
                          (~? (~@ #:interval interval))))

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
                              [(not (unbox go?)) (unless (unbox stop?) (loop k))]
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

(define (call-with-workers n make-mix grace proc)
  (define go? (box #f))
  (define stop? (box #f))
  (define counts (make-vector n 0))
  (define failure (box #f))
  (define-values (custodian workers) (start-workers n go? stop? counts failure make-mix))
  (define (stop!)
    (set-box! stop? #t)
    (reap! custodian workers grace))
  (define result
    (with-handlers ([(lambda (_e) #t) (lambda (e) (stop!) (raise e))])
      (proc (lambda () (set-box! go? #t))
            (lambda () (for/sum ([k (in-vector counts)]) k)))))
  (define stopped (stop!))
  (match (unbox failure)
    [#f (values result stopped)]
    [e (raise e)]))

(define (scaling-run n state
                     #:seconds seconds
                     #:grace [grace 30]
                     #:mix [make-mix small-mix]
                     #:ops-per-mix [ops-per-mix ops-per-iteration]
                     #:probe [probe void]
                     #:tasks [tasks task-dir])
  (force collector-sweepers)
  (define baseline (settled-thread-count tasks))
  (define-values (run stopped)
    (call-with-workers
     n make-mix grace
     (lambda (go! mixes-done)
       (define-values (timed peak)
         (with-peak-sampler (#:tasks tasks)
           (define probe-start (probe))
           (define t0 (clock-ms))
           (go!)
           (define main-steps (run-main state (+ t0 (* 1000.0 seconds))))
           (define elapsed-s (/ (- (clock-ms) t0) 1000.0))
           (define ops (mixes-done))
           (hasheq 'ops_per_s (/ (* ops-per-mix ops) elapsed-s)
                   'mixes ops
                   'elapsed_s elapsed-s
                   'main_steps_per_s (/ main-steps elapsed-s)
                   'started_ms t0
                   'probe_start probe-start
                   'probe_stop (probe))))
       (hash-set* timed 'peak_os_threads peak 'baseline_os_threads baseline))))
  (hash-set run 'stopped stopped))

(define (scaling-cases #:seconds seconds
                       #:threads [threads thread-counts]
                       #:states [states main-states]
                       #:mix [mix 'small]
                       #:on-run [on-run void])
  (define chosen (mix-named mix))
  (define make-mix (op-mix-make chosen))
  (define ops-per-mix (op-mix-ops chosen))
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
           (only-in racket/port with-output-to-string)
           (only-in racket/runtime-path define-runtime-path)
           (only-in racket/system system*)
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
      (define-values (result peak)
        (with-peak-sampler (#:tasks dir #:interval 0.01)
          (touch! dir '("2" "3" "4"))
          (sleep 0.2)
          (for ([name (in-list '("2" "3" "4"))]) (delete-file (build-path dir name)))
          (sleep 0.2)
          'done))
      (check-equal? (list result peak) '(done 4)))
    (define-values (result peak) (with-peak-sampler (#:interval 0.01 #:tasks absent) 'done))
    (check-equal? (list result peak) '(done #f))
    (check-false (peak-of '(3 #f)))
    (check-equal? (peak-of '(3 5 4)) 5))

  (define (live-threads custodian)
    (for/list ([v (in-list (custodian-managed-list custodian (current-custodian)))]
               #:when (and (thread? v) (not (thread-dead? v))))
      v))

  (test-case "a raising probe stops the sampler and the workers"
    (call-with-process-deadline
     120
     (lambda ()
       (for ([failing-call (in-list '(1 2))])
         (define calls 0)
         (define (probe)
           (set! calls (add1 calls))
           (when (= calls failing-call) (raise-user-error 'probe "failed")))
         (define custodian (make-custodian))
         (define t0 (clock-ms))
         (parameterize ([current-custodian custodian])
           (check-exn #rx"probe: failed"
                      (lambda () (scaling-run 2 'idle #:seconds 0.2 #:grace 20 #:probe probe))))
         (check-equal? (live-threads custodian) '() (format "call ~a" failing-call))
         (check-true (< (- (clock-ms) t0) 10000.0) "no worker waited out the grace period")))))

  (test-case "a run's peak OS threads count a thread per worker over the tasks before it"
    (call-with-process-deadline
     120
     (lambda ()
       (define run (scaling-run 2 'idle #:seconds 0.3))
       (define peak (hash-ref run 'peak_os_threads))
       (define baseline (hash-ref run 'baseline_os_threads))
       (cond
         [(directory-exists? "/proc/self/task")
          (check-pred exact-positive-integer? baseline)
          (check-true (>= peak (+ baseline 2)) (format "~a" run))]
         [else (check-false peak) (check-false baseline)])
       (define elsewhere (scaling-run 1 'idle #:seconds 0.1 #:tasks absent))
       (check-false (hash-ref elsewhere 'peak_os_threads))
       (check-false (hash-ref elsewhere 'baseline_os_threads)))))

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

  (define-runtime-path harness-module "harness.rkt")
  (define-runtime-path threads-module "threads.rkt")

  (define (counts-under omp mixes rounds #:mkl [mkl #f])
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"OMP_NUM_THREADS" (string->bytes/utf-8 (number->string omp)))
    (environment-variables-set! env #"MKL_NUM_THREADS" (and mkl (string->bytes/utf-8 (number->string mkl))))
    (define program
      `(begin
         (require (only-in ffi/unsafe _fun _int get-ffi-obj)
                  (only-in torch/foreign/raw/native-library native-library)
                  (only-in (file ,(path->string harness-module)) measure)
                  (only-in (file ,(path->string threads-module)) scaling-cases))
         (define (worker-team)
           (define team (box #f))
           (define at-threads (get-ffi-obj "_ZN2at15get_num_threadsEv" native-library (_fun -> _int)))
           (thread-wait (thread (lambda () (set-box! team (at-threads))) #:pool 'own))
           (unbox team))
         (define (counts mix)
           (define runs '())
           (measure (scaling-cases #:seconds 0.5 #:threads '(2) #:states '(idle) #:mix mix
                                   #:on-run (lambda (_state _n run) (set! runs (cons run runs))))
                    #:warmup 0 #:rounds ,rounds #:load void)
           (for/list ([run (in-list (reverse runs))])
             (list (hash-ref run 'baseline_os_threads) (hash-ref run 'peak_os_threads))))
         (write (cons (worker-team) (map counts ',mixes)))))
    (define ok? #f)
    (define out
      (parameterize ([current-environment-variables env])
        (with-output-to-string
          (lambda ()
            (set! ok? (system* (find-executable-path (find-system-path 'exec-file))
                               "-l" "racket/base" "-e" (format "~s" program)))))))
    (check-true ok? out)
    (read (open-input-string out)))

  (when (directory-exists? "/proc/self/task")
    (test-case "a process's first run starts the collector sweepers a run's threads would"
      (call-with-process-deadline
       120
       (lambda ()
         (scaling-run 1 'idle #:seconds 0.1)
         (define before (settled-thread-count task-dir))
         (collect-with-allocating-threads! (apply max thread-counts))
         (check-equal? (settled-thread-count task-dir) before))))

    (test-case "a run's peak counts each worker's intra-op team, as ATen sizes it"
      (call-with-process-deadline
       300
       (lambda ()
         (define (added omp #:mkl [mkl #f])
           (match-define (list team (list (list base peak))) (counts-under omp '(grain) 1 #:mkl mkl))
           (values team (- peak base)))
         (define-values (team₁ one) (added 1))
         (check-equal? team₁ 1)
         (for ([mkl (in-list '(#f 2))])
           (define-values (team four) (added 4 #:mkl mkl))
           (define seen (format "MKL ~a: team ~a, added ~a at OMP 1, ~a at OMP 4" mkl team one four))
           (if (< team 2)
               (printf "[threads] team check skipped: ~a\n" seen)
               (check-true (>= four (+ one (* 2 (sub1 team)))) seen))))))

    (test-case "back-to-back runs start from the same threads and peak alike"
      (call-with-process-deadline
       300
       (lambda ()
         (match-define (list _ (list (list base₁ peak₁) (list base₂ peak₂)))
           (counts-under 4 '(grain) 2))
         (define seen (format "~a then ~a" (list base₁ peak₁) (list base₂ peak₂)))
         (check-true (<= (abs (- base₂ base₁)) 1) seen)
         (check-true (<= (abs (- peak₂ peak₁)) 1) seen)))))

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
