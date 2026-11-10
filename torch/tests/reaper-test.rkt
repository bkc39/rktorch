#lang racket/base

(module+ test
  (require (only-in ffi/unsafe register-finalizer)
           (only-in ffi/unsafe/vm vm-eval)
           (only-in racket/file file->string)
           (only-in racket/match match)
           (only-in racket/os getpid)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../foreign/raw/cas.rkt" cas-add!)
           (only-in "../runtime/reaper.rkt"
                    drain-reapers! reap-when-dead! reaped reaper-count wake-reapers!)
           (only-in (submod "../runtime/reaper.rkt" private)
                    heap-object/c reaper-statistics reaper-threads)
           (only-in "private/wedge.rkt" exit-status-of)
           (only-in "private/within.rkt" within))

  (define (statistic key)
    (hash-ref (reaper-statistics) key))

  (define (with-reapers k thunk)
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"RKTORCH_REAPERS" (string->bytes/utf-8 (number->string k)))
    (parameterize ([current-environment-variables env])
      (thunk)))

  (define exactly-once-program
    '(begin
       (require (only-in torch/foreign/raw/cas cas-add!)
                (only-in torch/runtime/reaper
                         drain-reapers! reap-when-dead! reaped reaper-count wake-reapers!)
                (only-in (submod torch/runtime/reaper private) reaper-statistics)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define n 200000)
       (define counts (for/vector #:length n ([_ (in-range n)]) (box 0)))
       (define kind (reaped 'counted (lambda (v) (cas-add! (vector-ref counts (vector-ref v 0)) 1))))
       (define (released) (hash-ref (reaper-statistics) 'released))
       (call-with-process-deadline
        100
        (lambda ()
          (for ([i (in-range n)]) (reap-when-dead! kind (vector i)))
          (collect-garbage)
          (wake-reapers!)
          (let wait ([seen -1])
            (sleep 0.05)
            (define now (released))
            (unless (= now seen) (wait now)))
          (define drained? (drain-reapers! #:deadline-ms 60000))
          (define once (for/sum ([c (in-vector counts)]) (if (= 1 (unbox c)) 1 0)))
          (eprintf "reapers ~a, drained ~a, released once ~a of ~a\n" reaper-count drained? once n)
          (exit (if (and drained? (= reaper-count 4) (= once n)) 0 1))))))

  (test-case "L1: four reapers release each of 200k objects exactly once"
    (check-equal? (with-reapers 4 (lambda () (exit-status-of exactly-once-program #:within 120)))
                  0))

  (define objects 1000)

  (test-case "L2: an object a pending will still reaches is released only after the will"
    (within
     60
     (lambda ()
       (define released (box 0))
       (define early (box 0))
       (define wills (box 0))
       (define kind (reaped 'ordered (lambda (v)
                                        (vector-set! v 1 'released)
                                        (cas-add! released 1))))
       (for ([i (in-range objects)])
         (define obj (make-vector 2 i))
         (reap-when-dead! kind obj)
         (register-finalizer (vector obj)
                             (lambda (u)
                               (cas-add! wills 1)
                               (when (eq? 'released (vector-ref (vector-ref u 0) 1))
                                 (cas-add! early 1)))))
       (collect-garbage)
       (check-true (drain-reapers!))
       (check-equal? (unbox released) 0 "released while a will still reached it")
       (let settle ([k 0])
         (sync/timeout 0.05 never-evt)
         (collect-garbage)
         (drain-reapers!)
         (when (and (< k 40) (< (unbox released) objects))
           (settle (add1 k))))
       (check-equal? (unbox wills) objects)
       (check-equal? (unbox early) 0)
       (check-equal? (unbox released) objects))))

  (test-case "L3: an ordered guardian never reports an object its representative reaches"
    (within
     60
     (lambda ()
       (define g (vm-eval '(make-guardian #t)))
       (for ([i (in-range objects)])
         (define obj (make-vector 2 i))
         (g obj (lambda () (vector-ref obj 0))))
       (for ([_ (in-range 3)]) (collect-garbage))
       (check-equal? (let loop ([k 0]) (if (g) (loop (add1 k)) k)) 0)
       (define released (box 0))
       (define kind (reaped 'plain (lambda (_v) (cas-add! released 1))))
       (for ([i (in-range objects)])
         (reap-when-dead! kind (make-vector 2 i)))
       (collect-garbage)
       (check-true (drain-reapers!))
       (check-equal? (unbox released) objects))))

  (test-case "reaper-count starts K reapers, each alive and of its own pool"
    (check-equal? (length (reaper-threads)) reaper-count)
    (check-true (andmap (lambda (t) (not (thread-dead? t))) (reaper-threads)))
    (check-true (heap-object/c (vector 1)))
    (check-false (heap-object/c 7)))

  (define (main-cpu-ms)
    (define stat (file->string (format "/proc/self/task/~a/stat" (getpid))))
    (match (regexp-match #px"\\) (?:\\S+ ){11}(\\d+) (\\d+) " stat)
      [(list _ utime stime) (* 10 (+ (string->number utime) (string->number stime)))]))

  (define wake-count 1000)

  (define (wake-parked-reapers!)
    (let loop ([woken 0])
      (when (< woken wake-count)
        (define before (statistic 'wakes))
        (wake-reapers!)
        (define deadline (+ (current-inexact-milliseconds) 5))
        (let spin ()
          (when (and (= before (statistic 'wakes)) (< (current-inexact-milliseconds) deadline))
            (spin)))
        (loop (if (> (statistic 'wakes) before) (add1 woken) woken)))))

  (test-case "L10: a thousand wakes of idle reapers cost main at most 20 ms of CPU"
    (cond
      [(not (file-exists? (format "/proc/self/task/~a/stat" (getpid))))
       (displayln "skipped: no /proc/self/task on this host")]
      [else
       (within
        120
        (lambda ()
          (define before (main-cpu-ms))
          (thread-wait (thread wake-parked-reapers! #:pool 'own))
          (define spent (- (main-cpu-ms) before))
          (check-true (<= spent 20) (format "main spent ~a ms of CPU" spent))))]))

  (define-runtime-path place-releases "private/place-releases.rkt")

  (define place-releases-program
    `(begin
       (require (only-in racket/place dynamic-place place-dead-evt))
       (define p (dynamic-place (string->path ,(path->string place-releases)) 'main))
       (define answer (sync/timeout 60 p))
       (define dead (sync/timeout 10 (place-dead-evt p)))
       (eprintf "place: ~s, dead ~a\n" answer (and dead #t))
       (exit (if (and (equal? answer '(#t #t #t #t #t)) dead) 0 1))))

  (test-case "a place other than the main one starts no reapers and still releases its tensors"
    (check-equal? (exit-status-of place-releases-program #:within 120) 0
                  (string-append "(no reaper threads?, drained?, back to baseline?,"
                                 " a run per tensor?, every release on the place's main OS thread?)")))

  (define both-wake-program
    '(begin
       (require (only-in racket/list remove-duplicates)
                (only-in torch/foreign/raw/cas cas-claim! cas-update!)
                (only-in torch/foreign/raw/race-points race-hook)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define arrivals (box '()))
       (define woken (box '()))
       (define armed (box #f))
       (define held (box #f))
       (define resumed (box #f))
       (set-box! race-hook
                 (lambda (name subject)
                   (case name
                     [(reaper-parking) (cas-update! arrivals (lambda (ts) (cons subject ts)))]
                     [(reaper-woken)
                      (when (unbox armed)
                        (cas-update! woken (lambda (ts) (cons subject ts)))
                        (when (cas-claim! held #f subject #:site 'gate)
                          (let spin () (unless (unbox resumed) (spin)))))]
                     [else (void)])))
       (define (distinct b)
         (length (remove-duplicates (unbox b) eq?)))
       (define (await k b)
         (sync/timeout 10 (thread (lambda ()
                                    (let wait ()
                                      (when (< (distinct b) k)
                                        (sleep 0.001)
                                        (wait)))))))
       (define wake-reapers! (dynamic-require 'torch/runtime/reaper 'wake-reapers!))
       (call-with-process-deadline
        50
        (lambda ()
          (await 2 arrivals)
          (set-box! arrivals '())
          (set-box! armed #t)
          (wake-reapers!)
          (await 1 woken)
          (wake-reapers!)
          (await 2 woken)
          (define both-woken (distinct woken))
          (set-box! resumed #t)
          (await 2 arrivals)
          (eprintf "reapers woken while the first stayed awake: ~a, parked again: ~a\n"
                   both-woken (distinct arrivals))
          (exit (if (= 2 both-woken (distinct arrivals)) 0 1))))))

  (test-case "a second wake reaches the other parked reaper while the first one woken is awake"
    (check-equal? (with-reapers 2 (lambda () (exit-status-of both-wake-program #:within 60)))
                  0))

  (define flushed-program
    '(begin
       (require (only-in torch + native-memory-use sum zeros)
                (only-in (submod torch/runtime/reaper private) reaper-statistics reaper-threads)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define (cpu-bytes)
         (for/sum ([entry (in-list (native-memory-use))]) (cdr entry)))
       (define (churn!)
         (for ([_ (in-range 2000)])
           (sum (+ (zeros 64 64) 1.0))))
       (define (released)
         (hash-ref (reaper-statistics) 'released))
       (call-with-process-deadline
        50
        (lambda ()
          (zeros 1)
          (define baseline (cpu-bytes))
          (plumber-flush-all (current-plumber))
          (define running? (not (ormap thread-dead? (reaper-threads))))
          (define before (released))
          (for-each thread-wait (for/list ([_ (in-range 2)]) (thread churn! #:pool 'own)))
          (collect-garbage)
          (zeros 1)
          (define deadline (+ (current-inexact-milliseconds) 10000))
          (define back?
            (let wait ()
              (cond
                [(<= (cpu-bytes) (+ baseline (* 2 64 64 4))) #t]
                [(> (current-inexact-milliseconds) deadline) #f]
                [else (sleep 0.01) (wait)])))
          (eprintf "reapers running ~a, released ~a, back ~a (baseline ~a, now ~a)\n"
                   running? (- (released) before) back? baseline (cpu-bytes))
          (exit (if (and running? back? (> (released) before)) 0 1))))))

  (test-case "a plumber-flush-all in a live program leaves the reapers reclaiming"
    (check-equal? (exit-status-of flushed-program #:within 60) 0))

  (define held-reaper-program
    '(begin
       (require (only-in torch zeros)
                (only-in torch/foreign/raw/cas cas-add! cas-claim!)
                (only-in torch/foreign/raw/race-points race-hook)
                (only-in torch/runtime/reaper reap-when-dead! reaped wake-reapers!)
                (only-in (submod torch/runtime/reaper private) reaper-threads)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define holding (box #f))
       (define go (box #f))
       (set-box! race-hook
                 (lambda (name subject)
                   (when (and (eq? name 'reaper-taken)
                              (memq subject (reaper-threads))
                              (cas-claim! holding #f subject #:site 'gate))
                     (let spin () (unless (unbox go) (spin))))))
       (define released (box 0))
       (define kind (reaped 'counted (lambda (_v) (cas-add! released 1))))
       (define (drop-and-collect! n)
         (for ([_ (in-range n)]) (reap-when-dead! kind (make-vector 1 0)))
         (collect-garbage))
       (call-with-process-deadline
        50
        (lambda ()
          (zeros 1)
          (let hold ()
            (unless (unbox holding)
              (drop-and-collect! 1)
              (wake-reapers!)
              (sleep 0.01)
              (hold)))
          (define before (unbox released))
          (drop-and-collect! 5)
          (zeros 1)
          (define by-the-op (- (unbox released) before))
          (set-box! go #t)
          (eprintf "released by the op after the collection: ~a of 5\n" by-the-op)
          (exit (if (= by-the-op 5) 0 1))))))

  (test-case "with its one reaper held mid-batch, the first op after a collection releases what it found"
    (check-equal? (with-reapers 1 (lambda () (exit-status-of held-reaper-program #:within 60))) 0))

  (define collecting-op-program
    '(begin
       (require (only-in ffi/unsafe/vm vm-eval)
                (only-in torch zeros)
                (only-in torch/foreign/raw/cas cas-add! cas-claim!)
                (only-in torch/foreign/raw/race-points race-hook)
                (only-in torch/runtime/reaper reap-when-dead! reaped wake-reapers!)
                (only-in (submod torch/runtime/reaper private) reaper-threads)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define collections (vm-eval '(lambda () (collections))))
       (define holding (box #f))
       (define go (box #f))
       (set-box! race-hook
                 (lambda (name subject)
                   (when (and (eq? name 'reaper-taken)
                              (memq subject (reaper-threads))
                              (cas-claim! holding #f subject #:site 'gate))
                     (let spin () (unless (unbox go) (spin))))))
       (define registered (box 0))
       (define released (box 0))
       (define kind (reaped 'counted (lambda (_v) (cas-add! released 1))))
       (define (drop! n)
         (for ([_ (in-range n)]) (reap-when-dead! kind (make-vector 1 0)))
         (set-box! registered (+ (unbox registered) n)))
       (define (collecting-op-leaves held)
         (drop! 5)
         (define before (collections))
         (zeros (* 2 1024 1024))
         (and (> (collections) before) (- (unbox registered) (unbox released) held)))
       (call-with-process-deadline
        50
        (lambda ()
          (let hold ()
            (unless (unbox holding)
              (drop! 1)
              (collect-garbage)
              (wake-reapers!)
              (sleep 0.01)
              (hold)))
          (zeros 1)
          (define held (- (unbox registered) (unbox released)))
          (define left
            (filter values (for/list ([_ (in-range 400)]) (collecting-op-leaves held))))
          (set-box! go #t)
          (eprintf "unreleased after each of ~a ops that collected: ~s\n" (length left) left)
          (exit (if (and (>= (length left) 20) (andmap zero? left)) 0 1))))))

  (test-case "with its one reaper held mid-batch, an op whose charge makes a collection due releases what that collection found before it returns"
    (check-equal? (with-reapers 1 (lambda () (exit-status-of collecting-op-program #:within 60))) 0))

  (define raise-under-the-take-lock-program
    '(begin
       (require (only-in torch/foreign/raw/regions with-uninterruptible)
                (only-in (submod torch/runtime/reaper private) with-take-lock)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (call-with-process-deadline
        30
        (lambda ()
          (define raised
            (with-handlers ([symbol? values])
              (with-uninterruptible
                (with-take-lock
                  (raise 'inside)))))
          (define other
            (thread (lambda () (with-uninterruptible (with-take-lock (void)))) #:pool 'own))
          (define free? (and (sync/timeout 5 other) #t))
          (eprintf "raised ~s, another thread then took the lock ~a\n" raised free?)
          (exit (if (and (eq? raised 'inside) free?) 0 1))))))

  (test-case "a raise under the take lock leaves the lock free"
    (check-equal? (exit-status-of raise-under-the-take-lock-program #:within 60) 0))

  (test-case "L15: a process exits with status 0 while its reapers are parked"
    (check-equal? (exit-status-of '(begin
                                     (require (only-in torch zeros))
                                     (zeros 2 2)
                                     (collect-garbage)
                                     (sleep 0.2))
                                  #:within 60)
                  0)))
