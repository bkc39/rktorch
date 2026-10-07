#lang racket/base

;; #262 question 4: is libtorch's intra-op thread count per OS thread, does
;; at::set_num_threads from a worker change the trainer's, and does each OS
;; thread that runs a parallel op get its own OpenMP team?
;;
;; libtorch applies its thread count lazily, once per OS thread, so each
;; scenario runs in a fresh process.

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
         (only-in racket/format ~a ~s)
         (only-in racket/port port->string with-input-from-string)
         ;; whole-module require on purpose (only-in breaks its expansion)
         racket/runtime-path
         (only-in racket/system process*/ports)
         "common.rkt")

(define-runtime-path shim-path "shim.rkt")
(define-runtime-path common-path "common.rkt")

(define (run-scenario body)
  (define program
    `(begin
       (require (file ,(path->string shim-path))
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
         (define failure (box #f))
         (define workers
           (for/list ([_ (in-range 4)])
             (thread (lambda ()
                       (with-handlers ([(lambda (_) #t) (lambda (e) (set-box! failure e))])
                         (prepare)
                         (op))
                       (semaphore-post done)
                       (semaphore-wait release))
                     #:pool 'own)))
         (for ([_ (in-range 4)]) (semaphore-wait done))
         (define tasks (os-task-count))
         (for ([_ (in-range 4)]) (semaphore-post release))
         (for-each thread-wait workers)
         (when (unbox failure) (raise (unbox failure)))
         tasks)
       (define (team-scenario prepare)
         (define start (os-task-count))
         (parallel-add!)
         (define after-trainer (os-task-count))
         (list (cons "OS threads at start" start)
               (cons "after the trainer's parallel add" after-trainer)
               (cons "while 4 workers that ran a parallel add are parked"
                     (team-round prepare parallel-add!))
               (cons "while 4 workers that ran a 1024 matmul are parked"
                     (team-round prepare parallel-matmul!))
               (cons "trainer at / omp / mkl" (counts))))
       (define (on-worker thunk)
         (define-values (_ms results) (run-parallel 1 (lambda (_i) (thunk))))
         (car results))
       (write (let () ,@body))))
  (define-values (out in _pid _err control)
    (apply values (process*/ports #f #f (current-error-port)
                                  (find-exe) "-l" "racket/base" "-e" (~s program))))
  (close-output-port in)
  (define text (port->string out))
  (control 'wait)
  (close-input-port out)
  (define result (with-input-from-string text read))
  (when (eof-object? result)
    (error 'run-scenario "the scenario exited ~a with no result" (control 'exit-code)))
  result)

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

;; The trainer's 1024 matmul, alone and while 4 workers loop matmuls of their
;; own at the default count or limited to 1. The trainer runs a parallel op
;; first, as a training step would, so its own count is fixed before any
;; worker sets one.
(define timing-scenario
  '((define (time-trainer reps)
      (define a (shim-randn 1024 1024))
      (shim-free (shim-matmul a a))
      (define start (current-inexact-monotonic-milliseconds))
      (for ([_ (in-range reps)]) (shim-free (shim-matmul a a)))
      (begin0 (/ (- (current-inexact-monotonic-milliseconds) start) reps)
              (shim-free a)))
    (define (with-busy-workers prepare thunk)
      (define stop (box #f))
      (define busy (make-semaphore 0))
      (define failure (box #f))
      (define (busy-loop)
        (prepare)
        (define a (shim-randn 512 512))
        (shim-free (shim-matmul a a))
        (semaphore-post busy)
        (let loop ()
          (shim-free (shim-matmul a a))
          (unless (unbox stop) (loop)))
        (shim-free a))
      (define workers
        (for/list ([_ (in-range 4)])
          (thread (lambda ()
                    (with-handlers ([(lambda (_) #t)
                                     (lambda (e)
                                       (set-box! failure e)
                                       (semaphore-post busy))])
                      (busy-loop)))
                  #:pool 'own)))
      (for ([_ (in-range 4)]) (semaphore-wait busy))
      (define result (and (not (unbox failure)) (thunk)))
      (set-box! stop #t)
      (for-each thread-wait workers)
      (when (unbox failure) (raise (unbox failure)))
      result)
    (parallel-add!)
    (define before (counts))
    (define alone (time-trainer 50))
    (define busy-default (with-busy-workers void (lambda () (time-trainer 50))))
    (define busy-omp
      (with-busy-workers (lambda () (omp-set-num-threads 1))
                         (lambda () (time-trainer 50))))
    (define busy-at
      (with-busy-workers (lambda () (at-set-num-threads 1))
                         (lambda () (time-trainer 50))))
    (list (cons "trainer at / omp / mkl before" before)
          (cons "trainer 1024 matmul ms, alone" (fmt-ms alone))
          (cons "with 4 workers looping 512 matmuls, default threads"
                (fmt-ms busy-default))
          (cons "with 4 workers that set omp_set_num_threads(1)" (fmt-ms busy-omp))
          (cons "with 4 workers that set at::set_num_threads(1)" (fmt-ms busy-at))
          (cons "trainer at / omp / mkl after" (counts))
          (cons "load" (load-average)))))

(module+ main
  (define shim-version
    (run-scenario '((shim-version))))
  (print-banner "Q4: libtorch intra-op threads from parallel threads"
                #:torch-version shim-version)
  (print-table-header '("scenario (fresh process)" "measurement" "value"))
  (for* ([s (in-list (append scenarios (list (cons "oversubscription timing" timing-scenario))))]
         [m (in-list (run-scenario (cdr s)))])
    (print-table-row (list (car s) (car m) (~a (cdr m)))))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
