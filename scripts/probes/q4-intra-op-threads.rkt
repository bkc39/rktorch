#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/format ~a)
         (only-in racket/match match-define)
         ;; whole-module require on purpose (only-in breaks its expansion)
         racket/runtime-path
         "common.rkt")

(define-runtime-path shim-path "shim.rkt")
(define-runtime-path common-path "common.rkt")

(define (run-scenario body)
  (define program
    `(begin
       (require racket/match
                syntax/parse/define
                (only-in threading ~>)
                (file ,(path->string shim-path))
                (file ,(path->string common-path)))
       (define elements (* 4 1024 1024))
       (define (parallel-add!)
         (define a (shim-randn elements))
         (define c (shim-add a a))
         (shim-free c)
         (shim-free a))
       (define (parallel-matmul!)
         (define a (shim-randn 1024 1024))
         (define c (shim-matmul a a))
         (shim-free c)
         (shim-free a))
       (define (counts)
         (list (at-get-num-threads) (omp-get-max-threads) (mkl-get-max-threads)))
       (define (team-round prepare op)
         (define done (make-semaphore 0))
         (define release (make-semaphore 0))
         (define-values (tasks _)
           (call-with-workers 4
                              (lambda (_i)
                                (with-final-post done (prepare) (op))
                                (semaphore-wait release))
                              (lambda (_workers)
                                (for ([_ (in-range 4)]) (semaphore-wait done))
                                (os-task-count))
                              #:release (lambda ()
                                          (for ([_ (in-range 4)]) (semaphore-post release)))))
         tasks)
       (define (tasks-back-to baseline)
         (let loop ([tries 0])
           (define now (os-task-count))
           (cond
             [(or (<= now baseline) (= tries 500)) now]
             [else (sleep 0.01) (loop (add1 tries))])))
       (define (team-counts prepare)
         (define start (os-task-count))
         (parallel-add!)
         (define after-trainer (os-task-count))
         (define add-round (team-round prepare parallel-add!))
         (define before-matmul (tasks-back-to after-trainer))
         (list (cons "OS threads at start" start)
               (cons "after the trainer's parallel add" after-trainer)
               (cons "while 4 workers that ran a parallel add are parked" add-round)
               (cons "after those workers exit, before the matmul round" before-matmul)
               (cons "while 4 workers that ran a 1024 matmul are parked"
                     (team-round prepare parallel-matmul!))
               (cons "trainer at / omp / mkl" (counts))))
       (define (team-scenario prepare)
         (cond
           [(os-task-count) (team-counts prepare)]
           [else (list (cons "OS threads" "n/a: no /proc/self/task on this platform"))]))
       (define (on-worker thunk)
         (match-define (list result) (span-value (run-parallel 1 (lambda (_i) (thunk)))))
         result)
       (write (let () ,@body))))
  (run-fresh-racket program))

(define scenarios
  (list
   (cons "defaults, nothing set"
         '((list (cons "trainer (main OS thread): at / omp / mkl" (counts))
                 (cons "fresh worker: at / omp / mkl" (on-worker counts)))))
   (cons "worker sets 1 after the trainer's first parallel op"
         '((parallel-add!)
           (define before (counts))
           (define worker (on-worker (lambda () (at-set-num-threads 1) (counts))))
           (list (cons "trainer before" before)
                 (cons "the worker that set 1" worker)
                 (cons "trainer after" (counts))
                 (cons "a later fresh worker" (on-worker counts)))))
   (cons "worker sets 1 before the trainer's first parallel op"
         '((define worker (on-worker (lambda () (at-set-num-threads 1) (counts))))
           (parallel-add!)
           (list (cons "the worker that set 1" worker)
                 (cons "trainer, first queried after" (counts)))))
   (cons "trainer ran randn and a matmul, no at::parallel_for op, then a worker sets 1"
         '((parallel-matmul!)
           (define worker (on-worker (lambda () (at-set-num-threads 1) (counts))))
           (parallel-add!)
           (list (cons "the worker that set 1" worker)
                 (cons "trainer after its first parallel add" (counts)))))
   (cons "trainer calls at::get_num_threads first, then a worker sets 1"
         '((define pinned (at-get-num-threads))
           (define worker (on-worker (lambda () (at-set-num-threads 1) (counts))))
           (parallel-add!)
           (list (cons "trainer's count when it asked" pinned)
                 (cons "the worker that set 1" worker)
                 (cons "trainer after its first parallel add" (counts))
                 (cons "a later fresh worker" (on-worker counts)))))
   (cons "OpenMP teams: 4 workers at the default count"
         '((team-scenario void)))
   (cons "OpenMP teams: 4 workers that call at::set_num_threads(1) first"
         '((team-scenario (lambda () (at-set-num-threads 1)))))
   (cons "OpenMP teams: 4 workers that call omp_set_num_threads(1) first"
         '((team-scenario (lambda () (omp-set-num-threads 1)))))))

(define timing-repeats 3)

(define timing-scenario
  `((define (time-trainer reps)
      (define a (shim-randn 1024 1024))
      (shim-free (shim-matmul a a))
      (define ms (timed (lambda () (for ([_ (in-range reps)]) (shim-free (shim-matmul a a))))))
      (shim-free a)
      (/ ms reps))
    (define (call-with-busy-workers prepare thunk)
      (define stop (make-semaphore 0))
      (define busy (make-semaphore 0))
      (define (busy-loop)
        (define a
          (with-final-post busy
            (prepare)
            (define warm (shim-randn 512 512))
            (shim-free (shim-matmul warm warm))
            warm))
        (let loop ()
          (shim-free (shim-matmul a a))
          (unless (sync/timeout 0 (semaphore-peek-evt stop)) (loop)))
        (shim-free a))
      (define-values (result _)
        (call-with-workers 4
                           (lambda (_i) (busy-loop))
                           (lambda (_workers)
                             (for ([_ (in-range 4)]) (semaphore-wait busy))
                             (thunk))
                           #:release (lambda () (semaphore-post stop))))
      result)
    (define-syntax-parse-rule (with-busy-workers prepare:expr body:expr ...+)
      (call-with-busy-workers prepare (lambda () body ...)))
    (define (median-ms thunk)
      (~> (repeat-runs ,timing-repeats thunk) median-of (real->decimal-string 1)))
    (parallel-add!)
    (define before (counts))
    (define alone (median-ms (lambda () (time-trainer 50))))
    (define busy-default
      (median-ms (lambda () (with-busy-workers void (time-trainer 50)))))
    (define busy-omp
      (median-ms (lambda ()
                   (with-busy-workers (lambda () (omp-set-num-threads 1)) (time-trainer 50)))))
    (define busy-at
      (median-ms (lambda ()
                   (with-busy-workers (lambda () (at-set-num-threads 1)) (time-trainer 50)))))
    (list (cons "trainer at / omp / mkl before" before)
          (cons ,(format "trainer 1024 matmul ms, alone (median of ~a)" timing-repeats) alone)
          (cons "with 4 workers looping 512 matmuls, default threads" busy-default)
          (cons "with 4 workers that set omp_set_num_threads(1)" busy-omp)
          (cons "with 4 workers that set at::set_num_threads(1)" busy-at)
          (cons "trainer at / omp / mkl after" (counts))
          (cons "load" (load-average)))))

(module+ main
  (define shim-version
    (run-scenario '((shim-version))))
  (print-banner "Q4: libtorch intra-op threads from parallel threads"
                #:torch-version shim-version)
  (print-table-header '("scenario (fresh process)" "measurement" "value"))
  (for ([s (in-list (append scenarios (list (cons "oversubscription timing" timing-scenario))))])
    (match-define (cons scenario body) s)
    (for ([m (in-list (run-scenario body))])
      (match-define (cons what value) m)
      (print-table-row (list scenario what (~a value)))))
  (print-footer))
