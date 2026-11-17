#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file make-temporary-directory)
           (only-in racket/match exn:misc:match? match match-define)
           (only-in racket/port with-output-to-string)
           (only-in racket/runtime-path define-runtime-path)
           (only-in racket/system system*)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "harness.rkt" clock-ms measure result-name result-per-call with-temporary-path)
           (only-in "threads.rkt" mix-names peak-of scaling-cases scaling-run thread-counts)
           (only-in (submod "threads.rkt" private)
                    collect-with-allocating-threads!
                    os-thread-count
                    run-worker!
                    settled-thread-count
                    task-dir
                    with-peak-sampler))

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

  (test-case "a worker runs once if the window closed while it waited, never if the run aborted"
    (define (scripted seen)
      (define left seen)
      (impersonate-box (box 'wait)
                       (lambda (_b _v)
                         (match left
                           [(list now) now]
                           [(cons now more) (set! left more) now]))
                       (lambda (_b v) v)))
    (for ([seen (in-list '((wait wait stop) (wait abort) (go go stop) (stop)))]
          [runs (in-list '(1 0 2 1))])
      (define ran (box 0))
      (define counted (box 0))
      (define worker
        (thread (lambda ()
                  (run-worker! (scripted seen)
                               (lambda () (set-box! ran (add1 (unbox ran))))
                               (lambda (k) (set-box! counted k))))))
      (check-not-false (sync/timeout 5 worker) (format "~a: the worker stops" seen))
      (kill-thread worker)
      (check-equal? (unbox ran) runs (format "~a" seen))
      (check-equal? (unbox counted) runs (format "~a" seen))))

  (test-case "a run that fails before the clock starts runs no mix"
    (call-with-process-deadline
     120
     (lambda ()
       (define ticks (box 0))
       (define (ticking)
         (lambda () (let loop () (unless (box-cas! ticks (unbox ticks) (add1 (unbox ticks))) (loop)))))
       (check-exn #rx"probe: down"
                  (lambda ()
                    (scaling-run 2 'idle #:seconds 0 #:grace 5 #:mix ticking
                                 #:probe (lambda () (raise-user-error 'probe "down")))))
       (check-equal? (unbox ticks) 0))))

  (define taskset (find-executable-path "taskset"))
  (define one-cpu
    (and (file-exists? "/proc/self/status")
         (match (call-with-input-file "/proc/self/status"
                  (lambda (in) (regexp-match #px"Cpus_allowed_list:\\s*(\\d+)" in)))
           [(list _ cpu) (bytes->string/utf-8 cpu)]
           [#f #f])))

  (when (and taskset one-cpu)
    (test-case "on one CPU, workers spinning when the clock starts still run the raising mix"
      (call-with-process-deadline
       120
       (lambda ()
         (define program
           `(begin
              (require (only-in (file ,(path->string threads-module)) scaling-run))
              (define (busy ms)
                (define until (+ (current-inexact-milliseconds) ms))
                (let loop () (when (< (current-inexact-milliseconds) until) (loop))))
              (write
               (for/sum ([_ (in-range 20)])
                 (define made (box 0))
                 (define probes (box 0))
                 (define (failing-first)
                   (set-box! made (add1 (unbox made)))
                   (if (= (unbox made) 1) (lambda () (raise-user-error 'mix "unsupported")) void))
                 (define (probe)
                   (set-box! probes (add1 (unbox probes)))
                   (when (= (unbox probes) 1) (busy 30)))
                 (with-handlers ([exn:fail:user? (lambda (_e) 0)])
                   (scaling-run 2 'idle #:seconds 0 #:mix failing-first #:probe probe)
                   1)))))
         (define ok? #f)
         (define out
           (with-output-to-string
             (lambda ()
               (set! ok? (system* taskset "-c" one-cpu
                                  (find-executable-path (find-system-path 'exec-file))
                                  "-l" "racket/base" "-e" (format "~s" program))))))
         (check-true ok? out)
         (check-equal? (read (open-input-string out)) 0 "runs that lost the worker's raise")))))

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
