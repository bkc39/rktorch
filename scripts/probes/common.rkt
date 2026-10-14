#lang racket/base

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
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
         (only-in racket/format ~a ~r ~s)
         (only-in racket/function const)
         (only-in racket/future processor-count)
         (only-in racket/list filter-not last)
         (only-in racket/math exact-round)
         (only-in racket/port port->string with-input-from-string)
         (only-in racket/promise delay force)
         (only-in racket/string string-join)
         (only-in racket/system process*/ports)
         ;; whole-module require: only-in strips the phase-1 syntax classes
         syntax/parse/define
         (only-in threading ~>))

(provide pthread-self
         main-os-thread
         load-average
         os-task-count
         print-banner
         print-footer
         print-table-header
         print-table-row
         with-atomic
         with-uninterruptible
         with-uninterruptible-lock
         with-final-post
         with-finalizer-in-flight
         cas-update!
         now-ms
         ms->seconds
         bytes->mib
         mib->bytes
         bytes->mib-string
         span-start
         span-end
         span-gc-ms
         span-counts
         span-value
         span-ms
         measure-span
         timed
         tally
         tally-open!
         tally-loop-for-seconds
         tally-close!
         tally-total
         tally-max-gap
         gap-ms
         gap->string
         call-with-workers
         run-fresh-racket
         run-parallel
         in-indexed
         main-mode-label
         main-mode-polls-scheduler?
         main-blocked
         main-polling
         main-busy
         foreign-modes
         main-modes
         repeat-runs
         median-of
         middle-by
         collect-past-canary!
         settle-wills!
         exn:fail:unsettled?
         with-unsettled-stop
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
            (~> (current-seconds) seconds->date (date->string #t))))
  (when torch-version
    (printf "- libtorch ~a\n" torch-version))
  (printf "- load average (1 min) at start: ~a\n\n" (load-average)))

(define (print-footer #:stopped [stopped #f])
  (define raised (unbox wills-raised))
  (when stopped
    (printf "\nstopped: ~a\n" stopped))
  (printf "\nload average (1 min) at end: ~a\n" (load-average))
  (when (pair? raised)
    (printf "wills that raised while settling: ~a (first: ~a)\n"
            (length raised) (last raised))))

(define (row->line cells)
  (string-join (map ~a cells) " | " #:before-first "| " #:after-last " |"))

(define (print-table-header header)
  (displayln (row->line header))
  (~> header (map (const "---") _) row->line displayln)
  (flush-output))

(define (print-table-row row)
  (displayln (row->line row))
  (flush-output))

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

(define-syntax-parse-rule (with-finally after:expr body:expr ...+)
  (dynamic-wind void (lambda () body ...) (lambda () after)))

(define-syntax-parse-rule (with-final-post sema:expr body:expr ...+)
  (let ([posted sema])
    (with-finally (semaphore-post posted) body ...)))

(define finalizers-in-flight (box 0))
(define finalizers-exited (box 0))

(define (finalizer-exited!)
  (cas-update! finalizers-exited add1)
  (cas-update! finalizers-in-flight sub1))

(define-syntax-parse-rule (with-finalizer-in-flight body:expr ...+)
  (begin
    (cas-update! finalizers-in-flight add1)
    (with-finally (finalizer-exited!) body ...)))

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

(define (bytes->mib n-bytes)
  (/ n-bytes bytes-per-mib 1.0))

(define (mib->bytes mib)
  (* mib bytes-per-mib))

(define (bytes->mib-string n-bytes)
  (~> n-bytes bytes->mib (real->decimal-string 1)))

(define gc-wall-ms
  (vm-eval '(lambda ()
              (let ([t (sstats-gc-real (statistics))])
                (+ (* 1000. (time-second t)) (/ (time-nanosecond t) 1e6))))))

(struct span (start end gc-ms counts value))

(define (span-ms s)
  (- (span-end s) (span-start s)))

(define (read-counters counters)
  (map (lambda (counter) (counter)) counters))

(define (measure-span thunk #:counters [counters '()])
  (define counts-start (read-counters counters))
  (define gc-start (gc-wall-ms))
  (define start (now-ms))
  (define value (thunk))
  (define end (now-ms))
  (span start end (- (gc-wall-ms) gc-start) (map - (read-counters counters) counts-start) value))

(define (timed thunk)
  (span-ms (measure-span thunk)))

(define (loop-for-seconds seconds step #:stop? [stop? (const #f)])
  (define deadline (+ (now-ms) (seconds->ms seconds)))
  (let loop ([done 0])
    (cond
      [(> (now-ms) deadline) done]
      [(stop? done) done]
      [else (step done) (loop (add1 done))])))

(struct lane (steps mark longest open-gap closed-at) #:mutable)

(define (tally n)
  (define start (now-ms))
  (build-vector (max n 1) (lambda (_) (lane 0 start 0.0 0.0 +inf.0))))

(define (tally-open! t)
  (define now (now-ms))
  (for ([l (in-vector t)])
    (set-lane-mark! l now)))

(define (lane-record-gap! l now)
  (set-lane-longest! l (max (lane-longest l) (- now (lane-mark l))))
  (set-lane-mark! l now))

(define (tally-step! t i)
  (define l (vector-ref t i))
  (define now (now-ms))
  (set-lane-steps! l (add1 (lane-steps l)))
  (unless (> now (lane-closed-at l))
    (lane-record-gap! l now)))

(define (tally-loop-for-seconds t i seconds step #:stop? [stop? (const #f)])
  (begin0
    (loop-for-seconds seconds
                      (lambda (done)
                        (step done)
                        (tally-step! t i))
                      #:stop? stop?)
    (set-lane-mark! (vector-ref t i) +inf.0)))

(define (tally-close! t)
  (define now (now-ms))
  (for ([l (in-vector t)])
    (set-lane-closed-at! l now)
    (set-lane-open-gap! l (- now (lane-mark l)))))

(define (tally-total t)
  (for/sum ([l (in-vector t)]) (lane-steps l)))

(struct gap (ms open?))

(define (tally-max-gap t)
  (for/fold ([longest (gap 0.0 #f)])
            ([l (in-vector t)])
    (define closed (lane-longest l))
    (define open (lane-open-gap l))
    (cond
      [(> open (max closed (gap-ms longest))) (gap open #t)]
      [(> closed (gap-ms longest)) (gap closed #f)]
      [else longest])))

(define (gap->string g)
  (define ms (real->decimal-string (gap-ms g) 1))
  (cond
    [(gap-open? g) (format "≥ ~a" ms)]
    [else ms]))

(define (in-indexed v)
  (in-parallel (in-vector v) (in-naturals)))

(define (call-with-workers n work main #:pool [pool 'own] #:release [release void])
  (define results (make-vector n #f))
  (define failures (make-vector n #f))
  (define workers
    (for/list ([i (in-range n)])
      (thread (lambda ()
                (with-handlers ([exn:fail? (lambda (e) (vector-set! failures i e))])
                  (vector-set! results i (work i))))
              #:pool pool)))
  (define (release-and-wait!)
    (release)
    (for-each thread-wait workers))
  (define result (with-finally (release-and-wait!) (main workers)))
  (cond
    [(for/first ([e (in-vector failures)] #:when e) e) => raise]
    [else (values result (vector->list results))]))

(define (run-fresh-racket program)
  (define-values (out in _pid _err control)
    (~> program
        ~s
        (process*/ports #f #f (current-error-port) (find-exe) "-l" "racket/base" "-e" _)
        (apply values _)))
  (close-output-port in)
  (define text (port->string out))
  (control 'wait)
  (close-input-port out)
  (define result (with-input-from-string text read))
  (when (eof-object? result)
    (error 'run-fresh-racket "the program exited ~a with no result" (control 'exit-code)))
  result)

(struct main-mode (label polls-scheduler? wait))

(define main-blocked
  (main-mode "blocked in thread-wait"
             #t
             (lambda (workers _on-wake) (for-each thread-wait workers))))

(define (waking-mode label pause #:for [seconds +inf.0] #:polls-scheduler? [polls? #t])
  (main-mode label
             polls?
             (lambda (workers on-wake)
               (define deadline (+ (now-ms) (seconds->ms seconds)))
               (let loop ()
                 (unless (or (andmap thread-dead? workers) (> (now-ms) deadline))
                   (pause workers)
                   (on-wake)
                   (loop))))))

(define (main-polling ms)
  (waking-mode (format "polling every ~a ms" ms)
               (lambda (workers)
                 (sync/timeout (ms->seconds ms)
                               (~> workers
                                   (filter-not thread-dead? _)
                                   (map thread-dead-evt _)
                                   (apply choice-evt _))))))

(define busy-sink (box #f))
(define busy-wake-ms 1.0)

(define main-busy
  (waking-mode "busy allocating Racket vectors (Racket-only)"
               (lambda (_workers)
                 (loop-for-seconds (ms->seconds busy-wake-ms)
                                   (lambda (_i) (set-box! busy-sink (make-vector 64 #f)))))))

(define foreign-usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))
(define foreign-usleep/blocking (get-ffi-obj "usleep" #f (_fun #:blocking? #t _uint -> _int)))
(define foreign-call-us 5000)

(define (foreign-mode how call seconds #:polls-scheduler? polls?)
  (waking-mode (format "in ~a ms foreign calls (usleep, ~a) for ~a s"
                       (/ foreign-call-us 1000) how seconds)
               (lambda (_workers) (call))
               #:for seconds
               #:polls-scheduler? polls?))

(define (foreign-modes seconds)
  (list (foreign-mode "plain _fun, never yielding"
                      (lambda () (foreign-usleep foreign-call-us))
                      seconds
                      #:polls-scheduler? #f)
        (foreign-mode "plain _fun, (sleep 0) after each"
                      (lambda ()
                        (foreign-usleep foreign-call-us)
                        (sleep 0))
                      seconds
                      #:polls-scheduler? #t)
        (foreign-mode "_fun #:blocking? #t, never yielding"
                      (lambda () (foreign-usleep/blocking foreign-call-us))
                      seconds
                      #:polls-scheduler? #f)))

(define (main-modes foreign-seconds)
  (list* main-blocked (main-polling 10) main-busy (foreign-modes foreign-seconds)))

(define (run-parallel n body
                      #:pool [pool 'own]
                      #:setup [setup void]
                      #:counters [counters '()]
                      #:main [mode main-blocked]
                      #:on-start [on-start void]
                      #:on-wake [on-wake void]
                      #:on-end [on-end void]
                      #:stop [stop (box #f)])
  (define arrived (box 0))
  (define go (box #f))
  (define (release!)
    (set-box! go #t)
    (set-box! stop #t))
  (define (work i)
    (with-finally (cas-update! arrived add1) (setup))
    (let spin ()
      (unless (unbox go) (spin)))
    (body i))
  (define (await-arrivals)
    (unless (= (unbox arrived) n)
      (sleep 0)
      (await-arrivals)))
  (define (measure-workers workers)
    (await-arrivals)
    (measure-span (lambda ()
                    (on-start)
                    (set-box! go #t)
                    ((main-mode-wait mode) workers on-wake)
                    (on-end))
                  #:counters counters))
  (cond
    [(zero? n)
     (setup)
     (measure-span (lambda ()
                     (on-start)
                     (begin0 (list (body 0)) (on-end)))
                   #:counters counters)]
    [else
     (define-values (window results)
       (call-with-workers n work measure-workers #:pool pool #:release release!))
     (struct-copy span window [value results])]))

(define (repeat-runs n thunk)
  (build-list n (lambda (_) (thunk))))

(define (median-of xs)
  (median < xs))

(define (middle-by key xs)
  (list-ref (sort xs < #:key key) (quotient (length xs) 2)))

(define (rate->string per-second)
  (~a (exact-round per-second)))

(define canary-timeout-seconds 0.5)

(define finalizer-executor
  (delay
    (let ([closure ((vm-primitive 'inspect/object) register-finalizer)])
      (or (for*/first ([k (in-range (closure 'length))]
                       [v (in-value (((closure 'ref k) 'ref) 'value))]
                       #:when (will-executor? v))
            v)
          (error 'probes "register-finalizer closes over no will executor on this VM")))))

(define no-will (string->uninterned-symbol "no-will"))

(define wills-raised (box '()))

(define (try-will executor)
  (with-handlers ([exn:fail? (lambda (e)
                               (set-box! wills-raised (cons (exn-message e) (unbox wills-raised)))
                               e)])
    (will-try-execute executor no-will)))

(define (run-ready-wills! executor)
  (let loop ([ran 0])
    (cond
      [(eq? (try-will executor) no-will) ran]
      [else (loop (add1 ran))])))

(define (collect-and-run-wills!)
  (begin0
    (with-atomic
      (collect-garbage 'major)
      (run-ready-wills! (force finalizer-executor)))
    (sleep 0)))

(struct exn:fail:unsettled exn:fail ())

(define unsettled-contexts (make-parameter '()))

(define (raise-unsettled reason)
  (raise (exn:fail:unsettled (~> (unsettled-contexts) (cons reason _) reverse (string-join ": "))
                             (current-continuation-marks))))

(define (call-with-unsettled-stop context thunk)
  (define outer (unsettled-contexts))
  (define (run)
    (parameterize ([unsettled-contexts (cons context outer)])
      (define reason (thunk))
      (and reason (raise-unsettled reason))))
  (cond
    [(pair? outer) (run)]
    [else (with-handlers ([exn:fail:unsettled? exn-message]) (run))]))

(define-syntax-parse-rule (with-unsettled-stop context:expr body:expr ...+)
  (call-with-unsettled-stop context (lambda () body ...)))

(define (collect-past-canary!)
  (define executor (force finalizer-executor))
  (define canary-finalized (make-semaphore 0))
  (define drained (box #f))
  (register-finalizer (box 0)
                      (lambda (_)
                        (with-atomic
                          (set-box! drained (run-ready-wills! executor))
                          (semaphore-post canary-finalized))))
  (collect-garbage 'major)
  (unless (sync/timeout canary-timeout-seconds canary-finalized)
    (raise-unsettled
     (format "a will held the finalizer thread past the ~a s canary" canary-timeout-seconds)))
  (unbox drained))

(define (collect-round!)
  (define now (collect-and-run-wills!))
  (cond
    [(positive? now) (values now 1)]
    [else (values (collect-past-canary!) 2)]))

(define (settle-wills! #:limit [limit 50])
  (let loop ([collections 0] [ran 0])
    (define exited (unbox finalizers-exited))
    (define-values (now round-collections) (collect-round!))
    (define so-far (+ collections round-collections))
    (cond
      [(and (zero? now) (= exited (unbox finalizers-exited)) (zero? (unbox finalizers-in-flight)))
       (values ran so-far)]
      [(>= so-far limit) (values #f so-far)]
      [else (loop so-far (+ ran now))])))
