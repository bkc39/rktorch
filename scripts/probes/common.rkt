#lang racket/base

(require (only-in ffi/unsafe
                  _double
                  _fun
                  _int
                  _pointer
                  _uintptr
                  get-ffi-obj
                  malloc
                  ptr-ref
                  register-finalizer)
         (only-in racket/date date->string date-display-format)
         (only-in racket/format ~a ~r)
         (only-in racket/future processor-count)
         (only-in racket/math exact-round)
         (only-in racket/string string-join))

(provide pthread-self
         main-os-thread
         on-main-os-thread?
         os-thread-label
         load-average
         os-task-count
         print-banner
         print-table
         print-table-header
         print-table-row
         run-parallel
         run-parallel/gc
         collect-past-canary!
         fmt-ms
         fmt-rate)

(define pthread-self (get-ffi-obj "pthread_self" #f (_fun -> _uintptr)))

(define main-os-thread (pthread-self))

(define (on-main-os-thread?)
  (= (pthread-self) main-os-thread))

(define (os-thread-label id)
  (if (= id main-os-thread) "main" "own"))

(define getloadavg (get-ffi-obj "getloadavg" #f (_fun _pointer _int -> _int)))

(define (load-average)
  (define out (malloc 3 _double 'atomic-interior))
  (cond
    [(< (getloadavg out 3) 1) "unknown"]
    [else (~r (ptr-ref out _double 0) #:precision 2)]))

(define (os-task-count)
  (define tasks "/proc/self/task")
  (and (directory-exists? tasks) (length (directory-list tasks))))

(define (print-banner title #:torch-version [torch-version #f])
  (printf "## ~a\n\n" title)
  (printf "- Racket ~a [~a], ~a cores, ~a\n"
          (version) (system-type 'vm) (processor-count)
          (parameterize ([date-display-format 'iso-8601])
            (date->string (seconds->date (current-seconds)) #t)))
  (when torch-version
    (printf "- libtorch ~a\n" torch-version))
  (printf "- load average (1 min) at start: ~a\n\n" (load-average)))

(define (row->line cells)
  (string-join (map ~a cells) " | " #:before-first "| " #:after-last " |"))

(define (print-table-header header)
  (displayln (row->line header))
  (displayln (row->line (map (lambda (_) "---") header)))
  (flush-output))

(define (print-table-row row)
  (displayln (row->line row))
  (flush-output))

(define (print-table header rows)
  (print-table-header header)
  (for-each print-table-row rows)
  (newline))

(define (run-parallel/gc n body
                         #:pool [pool 'own]
                         #:setup [setup void]
                         #:counter [counter (lambda () 0)])
  (define ready (make-semaphore 0))
  (define gate (make-semaphore 0))
  (define results (make-vector n #f))
  (define failures (make-vector n #f))
  (define (succeeded? i thunk)
    (with-handlers ([(lambda (_) #t) (lambda (e) (vector-set! failures i e) #f)])
      (thunk)
      #t))
  (define workers
    (for/list ([i (in-range n)])
      (thread (lambda ()
                (define set-up? (succeeded? i setup))
                (semaphore-post ready)
                (semaphore-wait gate)
                (when set-up?
                  (succeeded? i (lambda () (vector-set! results i (body i))))))
              #:pool pool)))
  (for ([_ (in-range n)]) (semaphore-wait ready))
  (define count-start (counter))
  (define gc-start (current-gc-milliseconds))
  (define start (current-inexact-monotonic-milliseconds))
  (for ([_ (in-range n)]) (semaphore-post gate))
  (for-each thread-wait workers)
  (define ms (- (current-inexact-monotonic-milliseconds) start))
  (define gc-ms (- (current-gc-milliseconds) gc-start))
  (define counted (- (counter) count-start))
  (define failure (for/first ([e (in-vector failures)] #:when e) e))
  (when failure (raise failure))
  (values ms (vector->list results) gc-ms counted))

(define (run-parallel n body #:pool [pool 'own])
  (define-values (ms results _gc-ms _counted) (run-parallel/gc n body #:pool pool))
  (values ms results))

(define (fmt-ms ms)
  (~r ms #:precision 1))

(define (fmt-rate per-second)
  (~a (exact-round per-second)))

(define canary-timeout-seconds 0.5)

(define (collect-past-canary!)
  (define canary-finalized (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post canary-finalized)))
  (collect-garbage 'major)
  (and (sync/timeout canary-timeout-seconds canary-finalized) #t))
