#lang racket/base

(require (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in torch + @ backward! randn relu sum tensor)
         (only-in racket/match match)
         (only-in torch/nn Linear Sequential cross-entropy parameters sgd step! zero-grads!)
         (only-in "harness.rkt" bench-case clock-ms))

(provide thread-counts
         main-states
         ops-per-iteration
         scaling-run
         scaling-cases)

(define thread-counts '(1 2 4 8))
(define main-states '(idle train))
(define ops-per-iteration 4)

(define (op-mix)
  (define a8 (randn 8 8))
  (define b8 (randn 8 8))
  (define a64 (randn 64 64))
  (define b64 (randn 64 64))
  (lambda ()
    (+ a8 b8)
    (@ a64 b64)
    (relu a64)
    (sum b64)))

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

(define (start-workers n stop? counts make-mix)
  (define mixes (for/list ([_ (in-range n)]) (make-mix)))
  (define custodian (make-custodian-at-root))
  (values custodian
          (parameterize ([current-custodian custodian])
            (for/list ([mix (in-list mixes)] [i (in-naturals)])
              (thread (lambda ()
                        (let loop ([k 0])
                          (unless (unbox stop?)
                            (mix)
                            (vector-set! counts i (add1 k))
                            (loop (add1 k)))))
                      #:pool 'own)))))

(define (reap! custodian workers grace)
  (for/sum ([w (in-list workers)])
    (cond
      [(sync/timeout grace w) 1]
      [else (parameterize ([current-custodian custodian]) (kill-thread w)) 0])))

(define (scaling-run n state #:seconds seconds #:grace [grace 30] #:mix [make-mix op-mix])
  (define stop? (box #f))
  (define counts (make-vector n 0))
  (define-values (custodian workers) (start-workers n stop? counts make-mix))
  (define stopped #f)
  (define t0 (clock-ms))
  (define-values (main-steps elapsed-s ops)
    (dynamic-wind
     void
     (lambda ()
       (define steps (run-main state (+ t0 (* 1000.0 seconds))))
       (values steps
               (/ (- (clock-ms) t0) 1000.0)
               (for/sum ([k (in-vector counts)]) k)))
     (lambda ()
       (set-box! stop? #t)
       (set! stopped (reap! custodian workers grace)))))
  (hasheq 'ops_per_s (/ (* ops-per-iteration ops) elapsed-s)
          'main_steps_per_s (/ main-steps elapsed-s)
          'stopped stopped))

(define (scaling-cases #:seconds seconds
                       #:threads [threads thread-counts]
                       #:states [states main-states]
                       #:on-run [on-run void])
  (for*/list ([state (in-list states)] [n (in-list threads)])
    (bench-case (string->symbol (format "~a/~a" state n))
                (lambda (_reps)
                  (define run (scaling-run n state #:seconds seconds))
                  (on-run state n run)
                  (hash-ref run 'ops_per_s))
                #:self-timed? #t)))

(module+ test
  (require (only-in racket/match exn:misc:match? match-define)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "harness.rkt" measure result-name result-per-call))

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
       (for ([r (in-list results)])
         (match-define (list ops) (result-per-call r))
         (check-true (if (memq (result-name r) '(idle/1 idle/2)) (positive? ops) (>= ops 0))
                     (format "~a" (result-name r))))
       (for ([entry (in-list runs)])
         (match-define (list state n run) entry)
         (check-equal? (hash-ref run 'stopped) n)
         (check-equal? (zero? (hash-ref run 'main_steps_per_s)) (eq? state 'idle)))
       (check-equal? (length runs) 4))))

  (define (busy)
    (lambda () (for/fold ([x 0]) ([i (in-range 2000000)]) (+ x i))))

  (test-case "a worker killed after the grace period keeps the ops it counted"
    (call-with-process-deadline
     120
     (lambda ()
       (define run (scaling-run 1 'idle #:seconds 0.3 #:grace 0 #:mix busy))
       (check-true (positive? (hash-ref run 'ops_per_s)))
       (check-true (<= 0 (hash-ref run 'stopped) 1)))))

  (test-case "workers stop when the main loop raises"
    (call-with-process-deadline
     120
     (lambda ()
       (define ticks (box 0))
       (define (ticking)
         (lambda () (let loop () (unless (box-cas! ticks (unbox ticks) (add1 (unbox ticks))) (loop)))))
       (check-exn exn:misc:match? (lambda () (scaling-run 2 'bogus #:seconds 0.1 #:mix ticking)))
       (define seen (unbox ticks))
       (sync (alarm-evt (+ (clock-ms) 300)))
       (check-equal? (unbox ticks) seen "no worker ran after the raise")))))
