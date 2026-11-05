#lang racket/base

(module+ test
  (require (only-in ffi/unsafe register-finalizer)
           (only-in ffi/unsafe/vm vm-eval)
           (only-in racket/file file->string)
           (only-in racket/match match)
           (only-in racket/os getpid)
           (only-in racket/string string-split)
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

  (define n 1000)

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
       (for ([i (in-range n)])
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
         (when (and (< k 40) (< (unbox released) n))
           (settle (add1 k))))
       (check-equal? (unbox wills) n)
       (check-equal? (unbox early) 0)
       (check-equal? (unbox released) n))))

  (test-case "L3: an ordered guardian never reports an object its representative reaches"
    (within
     60
     (lambda ()
       (define g (vm-eval '(make-guardian #t)))
       (for ([i (in-range n)])
         (define obj (make-vector 2 i))
         (g obj (lambda () (vector-ref obj 0))))
       (for ([_ (in-range 3)]) (collect-garbage))
       (check-equal? (let loop ([k 0]) (if (g) (loop (add1 k)) k)) 0)
       (define released (box 0))
       (define kind (reaped 'plain (lambda (_v) (cas-add! released 1))))
       (for ([i (in-range n)])
         (reap-when-dead! kind (make-vector 2 i)))
       (collect-garbage)
       (check-true (drain-reapers!))
       (check-equal? (unbox released) n))))

  (test-case "reaper-count starts K reapers, each alive and of its own pool"
    (check-equal? (length (reaper-threads)) reaper-count)
    (check-true (andmap (lambda (t) (not (thread-dead? t))) (reaper-threads)))
    (check-true (heap-object/c (vector 1)))
    (check-false (heap-object/c 7)))

  (define (main-cpu-ms)
    (define stat (file->string (format "/proc/self/task/~a/stat" (getpid))))
    (match (regexp-match #px"\\) (.*)$" stat)
      [(list _ rest)
       (define fields (string-split rest))
       (* 10 (+ (string->number (list-ref fields 11)) (string->number (list-ref fields 12))))]))

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
       (printf "skipped: no /proc/self/task on this host\n")]
      [else
       (within
        120
        (lambda ()
          (define before (main-cpu-ms))
          (thread-wait (thread wake-parked-reapers! #:pool 'own))
          (define spent (- (main-cpu-ms) before))
          (check-true (<= spent 20) (format "main spent ~a ms of CPU" spent))))]))

  (test-case "L15: a process exits with status 0 while its reapers are parked"
    (check-equal? (exit-status-of '(begin
                                     (require (only-in torch zeros))
                                     (zeros 2 2)
                                     (collect-garbage)
                                     (sleep 0.2))
                                  #:within 60)
                  0)))
