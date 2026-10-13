#lang racket/base

(require (for-syntax racket/base)
         (only-in ffi/unsafe
                  _double
                  _fun
                  _int
                  _pointer
                  _uintptr
                  get-ffi-obj
                  malloc
                  ptr-ref
                  register-finalizer)
         (only-in ffi/unsafe/atomic
                  end-atomic
                  end-uninterruptible
                  start-atomic
                  start-uninterruptible
                  uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         (only-in math/statistics median)
         (only-in racket/date date->string date-display-format)
         (only-in racket/format ~a ~r)
         (only-in racket/future processor-count)
         (only-in racket/math exact-round)
         (only-in racket/string string-join)
         ;; whole-module require: only-in strips the phase-1 syntax classes
         syntax/parse/define)

(provide pthread-self
         main-os-thread
         on-main-os-thread?
         os-thread-label
         load-average
         os-task-count
         print-banner
         print-footer
         print-table
         print-table-header
         print-table-row
         with-atomic
         with-uninterruptible
         with-uninterruptible-lock
         with-final-post
         cas-update!
         now-ms
         (struct-out span)
         span-ms
         measure-span
         timed
         loop-for-seconds
         call-with-workers
         run-parallel
         run-parallel/gc
         (struct-out main-mode)
         main-blocked
         main-polling
         main-busy
         main-modes
         repeat-runs
         median-of
         median-of-runs
         collect-past-canary!
         start-finalizer-thread!
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

(define (print-footer)
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))

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

(define-syntax-parse-rule (with-atomic body:expr ...+)
  (dynamic-wind start-atomic (lambda () body ...) end-atomic))

(define-syntax-parse-rule (with-uninterruptible body:expr ...+)
  (dynamic-wind start-uninterruptible (lambda () body ...) end-uninterruptible))

(define-syntax-parse-rule (with-uninterruptible-lock lock:expr body:expr ...+)
  (let ([held-lock lock])
    (dynamic-wind
     (lambda () (uninterruptible-lock-acquire held-lock))
     (lambda () body ...)
     (lambda () (uninterruptible-lock-release held-lock)))))

(define-syntax-parse-rule (with-final-post sema:expr body:expr ...+)
  (let ([posted sema])
    (dynamic-wind void (lambda () body ...) (lambda () (semaphore-post posted)))))

(define (cas-retry b f)
  (let retry ([tries 0])
    (define old (unbox b))
    (cond
      [(box-cas! b old (f old)) tries]
      [else (retry (add1 tries))])))

(define cas-update!
  (case-lambda
    [(b f) (cas-retry b f)]
    [(b f arg) (cas-retry b (lambda (old) (f old arg)))]))

(define (now-ms)
  (current-inexact-monotonic-milliseconds))

(struct span (start end gc-ms counted value))

(define (span-ms s)
  (- (span-end s) (span-start s)))

(define (measure-span thunk #:counter [counter (lambda () 0)])
  (define count-start (counter))
  (define gc-start (current-gc-milliseconds))
  (define start (now-ms))
  (define value (thunk))
  (define end (now-ms))
  (span start end (- (current-gc-milliseconds) gc-start) (- (counter) count-start) value))

(define (timed thunk)
  (span-ms (measure-span thunk)))

(define (loop-for-seconds seconds step #:stop? [stop? (lambda (_done) #f)])
  (define deadline (+ (now-ms) (* 1000 seconds)))
  (let loop ([done 0])
    (cond
      [(> (now-ms) deadline) done]
      [(stop? done) done]
      [else (step) (loop (add1 done))])))

(define (call-with-workers n work main #:pool [pool 'own] #:release [release void])
  (define results (make-vector n #f))
  (define failures (make-vector n #f))
  (define workers
    (for/list ([i (in-range n)])
      (thread (lambda ()
                (with-handlers ([(lambda (_) #t) (lambda (e) (vector-set! failures i e))])
                  (vector-set! results i (work i))))
              #:pool pool)))
  (define result
    (dynamic-wind void
                  (lambda () (main workers))
                  (lambda ()
                    (release)
                    (for-each thread-wait workers))))
  (for ([e (in-vector failures)] #:when e)
    (raise e))
  (values result (vector->list results)))

(struct main-mode (label wait))

(define main-blocked
  (main-mode "blocked in thread-wait"
             (lambda (workers _on-wake) (for-each thread-wait workers))))

(define (main-polling ms)
  (main-mode (format "polling every ~a ms" ms)
             (lambda (workers on-wake)
               (let loop ()
                 (define running (findf (lambda (w) (not (thread-dead? w))) workers))
                 (cond
                   [(not running) (void)]
                   [else
                    (sync/timeout (/ ms 1000.0) (thread-dead-evt running))
                    (on-wake)
                    (loop)])))))

(define busy-sink (box #f))
(define busy-wake-ms 1.0)

(define main-busy
  (main-mode "busy allocating Racket vectors"
             (lambda (workers on-wake)
               (let loop ([woke (now-ms)])
                 (set-box! busy-sink (make-vector 64 #f))
                 (define now (now-ms))
                 (cond
                   [(andmap thread-dead? workers) (void)]
                   [(> (- now woke) busy-wake-ms) (on-wake) (loop now)]
                   [else (loop woke)])))))

(define main-modes (list main-blocked (main-polling 10) main-busy))

(define (run-parallel/gc n body
                         #:pool [pool 'own]
                         #:setup [setup void]
                         #:counter [counter (lambda () 0)]
                         #:main [mode main-blocked]
                         #:on-wake [on-wake void])
  (define ready (make-semaphore 0))
  (define gate (make-semaphore 0))
  (define (open-gate) (for ([_ (in-range n)]) (semaphore-post gate)))
  (define-values (window results)
    (call-with-workers n
                       (lambda (i)
                         (with-final-post ready (setup))
                         (semaphore-wait gate)
                         (body i))
                       (lambda (workers)
                         (for ([_ (in-range n)]) (semaphore-wait ready))
                         (measure-span (lambda ()
                                         (open-gate)
                                         ((main-mode-wait mode) workers on-wake))
                                       #:counter counter))
                       #:pool pool
                       #:release open-gate))
  (struct-copy span window [value results]))

(define (run-parallel n body #:pool [pool 'own] #:main [mode main-blocked])
  (define window (run-parallel/gc n body #:pool pool #:main mode))
  (values (span-ms window) (span-value window)))

(define (repeat-runs n thunk)
  (for/list ([_ (in-range n)]) (thunk)))

(define (median-of xs)
  (median < xs))

(define (median-of-runs n thunk)
  (median-of (repeat-runs n thunk)))

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

(define (start-finalizer-thread!)
  (void (collect-past-canary!)))
