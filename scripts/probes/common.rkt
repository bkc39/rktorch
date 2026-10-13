#lang racket/base

(require (for-syntax racket/base)
         (only-in ffi/unsafe
                  _double
                  _fun
                  _int
                  _pointer
                  _uint
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
         (only-in ffi/unsafe/vm vm-eval vm-primitive)
         (only-in math/statistics median)
         (only-in racket/date date->string date-display-format)
         (only-in racket/format ~a ~r)
         (only-in racket/function const)
         (only-in racket/future processor-count)
         (only-in racket/list filter-not)
         (only-in racket/math exact-round)
         (only-in racket/string string-join)
         ;; whole-module require: only-in strips the phase-1 syntax classes
         syntax/parse/define)

(provide pthread-self
         main-os-thread
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
         ms->seconds
         bytes->mib
         mib->bytes
         span-start
         span-end
         span-gc-ms
         span-counted
         span-value
         span-ms
         measure-span
         timed
         loop-for-seconds
         call-with-workers
         run-parallel
         main-mode-label
         main-blocked
         main-polling
         main-busy
         main-in-foreign
         main-modes
         repeat-runs
         median-of
         collect-past-canary!
         settle-wills!
         ms->string
         mib->string
         rate->string)

(define pthread-self (get-ffi-obj "pthread_self" #f (_fun -> _uintptr)))

(define main-os-thread (pthread-self))

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
  (displayln (row->line (map (const "---") header)))
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

(define (cas-update! b f . args)
  (let retry ([tries 0])
    (define old (unbox b))
    (cond
      [(box-cas! b old (apply f old args)) tries]
      [else (retry (add1 tries))])))

(define (now-ms)
  (current-inexact-monotonic-milliseconds))

(define (ms->seconds ms)
  (/ ms 1000.0))

(define (seconds->ms seconds)
  (* 1000.0 seconds))

(define bytes-per-mib (* 1024 1024))

(define (bytes->mib bytes)
  (/ bytes bytes-per-mib 1.0))

(define (mib->bytes mib)
  (* mib bytes-per-mib))

(define gc-wall-ms
  (vm-eval '(lambda ()
              (let ([t (sstats-gc-real (statistics))])
                (+ (* 1000. (time-second t)) (/ (time-nanosecond t) 1e6))))))

(struct span (start end gc-ms counted value))

(define (span-ms s)
  (- (span-end s) (span-start s)))

(define (measure-span thunk #:counter [counter (const 0)])
  (define count-start (counter))
  (define gc-start (gc-wall-ms))
  (define start (now-ms))
  (define value (thunk))
  (define end (now-ms))
  (span start end (- (gc-wall-ms) gc-start) (- (counter) count-start) value))

(define (timed thunk)
  (span-ms (measure-span thunk)))

(define (loop-for-seconds seconds step #:stop? [stop? (const #f)])
  (define deadline (+ (now-ms) (seconds->ms seconds)))
  (let loop ([done 0])
    (cond
      [(> (now-ms) deadline) done]
      [(stop? done) done]
      [else (step done) (loop (add1 done))])))

(define (call-with-workers n work main #:pool [pool 'own] #:release [release void])
  (define results (make-vector n #f))
  (define failures (make-vector n #f))
  (define workers
    (for/list ([i (in-range n)])
      (thread (lambda ()
                (with-handlers ([exn:fail? (lambda (e) (vector-set! failures i e))])
                  (vector-set! results i (work i))))
              #:pool pool)))
  (define result
    (dynamic-wind void
                  (lambda () (main workers))
                  (lambda ()
                    (release)
                    (for-each thread-wait workers))))
  (cond
    [(for/first ([e (in-vector failures)] #:when e) e) => raise]
    [else (values result (vector->list results))]))

(struct main-mode (label wait))

(define main-blocked
  (main-mode "blocked in thread-wait"
             (lambda (workers _on-wake) (for-each thread-wait workers))))

(define (waking-mode label pause #:for [seconds +inf.0])
  (main-mode label
             (lambda (workers on-wake)
               (define deadline (+ (now-ms) (seconds->ms seconds)))
               (let loop ()
                 (cond
                   [(andmap thread-dead? workers) (void)]
                   [(> (now-ms) deadline) (for-each thread-wait workers)]
                   [else (pause workers) (on-wake) (loop)])))))

(define (main-polling ms)
  (waking-mode (format "polling every ~a ms" ms)
               (lambda (workers)
                 (sync/timeout (ms->seconds ms)
                               (apply choice-evt (map thread-dead-evt
                                                      (filter-not thread-dead? workers)))))))

(define busy-sink (box #f))
(define busy-wake-ms 1.0)

(define main-busy
  (waking-mode "busy allocating Racket vectors (Racket-only)"
               (lambda (_workers)
                 (loop-for-seconds (ms->seconds busy-wake-ms)
                                   (lambda (_i) (set-box! busy-sink (make-vector 64 #f)))))))

(define foreign-usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))
(define foreign-call-us 5000)

(define (main-in-foreign seconds)
  (waking-mode (format "in ~a ms foreign calls (usleep, plain _fun) for ~a s, then blocked"
                       (/ foreign-call-us 1000) seconds)
               (lambda (_workers) (foreign-usleep foreign-call-us))
               #:for seconds))

(define (main-modes foreign-seconds)
  (list main-blocked (main-polling 10) main-busy (main-in-foreign foreign-seconds)))

(define (run-parallel n body
                      #:pool [pool 'own]
                      #:setup [setup void]
                      #:counter [counter (const 0)]
                      #:main [mode main-blocked]
                      #:on-wake [on-wake void])
  (define ready (make-semaphore 0))
  (define gate (make-semaphore 0))
  (define (open-gate) (for ([_ (in-range n)]) (semaphore-post gate)))
  (define (measure-workers)
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
  (cond
    [(zero? n)
     (setup)
     (measure-span (lambda () (list (body 0))) #:counter counter)]
    [else (measure-workers)]))

(define (repeat-runs n thunk)
  (build-list n (lambda (_) (thunk))))

(define (median-of xs)
  (median < xs))

(define (ms->string ms)
  (~r ms #:precision 1))

(define (mib->string mib)
  (~r mib #:precision 1))

(define (rate->string per-second)
  (~a (exact-round per-second)))

(define canary-timeout-seconds 0.5)

(define (collect-past-canary!)
  (define canary-finalized (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post canary-finalized)))
  (collect-garbage 'major)
  (and (sync/timeout canary-timeout-seconds canary-finalized) #t))

(define finalizer-executor
  (let ([closure ((vm-primitive 'inspect/object) register-finalizer)])
    (or (for*/first ([k (in-range (closure 'length))]
                     [v (in-value (((closure 'ref k) 'ref) 'value))]
                     #:when (will-executor? v))
          v)
        (error 'probes "register-finalizer closes over no will executor on this VM"))))

(define no-will (string->uninterned-symbol "no-will"))

(define (run-ready-wills!)
  (let loop ([ran 0])
    (cond
      [(eq? (will-try-execute finalizer-executor no-will) no-will) ran]
      [else (loop (add1 ran))])))

(define (collect-and-run-wills!)
  (begin0
    (with-atomic
      (collect-garbage 'major)
      (run-ready-wills!))
    (sleep 0)))

(define (settle-wills! #:limit [limit 50])
  (let loop ([rounds 1] [ran 0])
    (define now (collect-and-run-wills!))
    (cond
      [(zero? now) ran]
      [(= rounds limit) #f]
      [else (loop (add1 rounds) (+ ran now))])))
