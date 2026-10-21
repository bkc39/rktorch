#lang racket/base

(require (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in torch + @ backward! randn relu sum tensor)
         (only-in torch/nn Linear Sequential cross-entropy parameters sgd step! zero-grads!)
         (only-in "harness.rkt" bench-case))

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

(define (clock-ms) (current-inexact-monotonic-milliseconds))

(define (run-main state deadline)
  (case state
    [(idle) (sync (alarm-evt deadline #t)) 0]
    [(train)
     (define step! (training-step))
     (let loop ([steps 0])
       (cond
         [(>= (clock-ms) deadline) steps]
         [else (step!) (loop (add1 steps))]))]))

(define (start-workers n stop? counts)
  (define mixes (for/list ([_ (in-range n)]) (op-mix)))
  (parameterize ([current-custodian (make-custodian-at-root)])
    (for/list ([mix (in-list mixes)] [i (in-naturals)])
      (thread (lambda ()
                (let loop ([k 0])
                  (cond
                    [(unbox stop?) (vector-set! counts i k)]
                    [else (mix) (loop (add1 k))])))
              #:pool 'own))))

(define (scaling-run n state #:seconds seconds #:grace [grace 30])
  (define stop? (box #f))
  (define counts (make-vector n 0))
  (define workers (start-workers n stop? counts))
  (define t0 (clock-ms))
  (define main-steps (run-main state (+ t0 (* 1000.0 seconds))))
  (define elapsed-s (/ (- (clock-ms) t0) 1000.0))
  (set-box! stop? #t)
  (define stopped
    (for/sum ([w (in-list workers)])
      (cond
        [(sync/timeout grace w) 1]
        [else (kill-thread w) 0])))
  (hasheq 'ops_per_s (/ (* ops-per-iteration (for/sum ([k (in-vector counts)]) k)) elapsed-s)
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
  (require rackunit
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
         (check-true (positive? (car (result-per-call r))) (format "~a" (result-name r))))
       (for ([entry (in-list runs)])
         (define run (caddr entry))
         (check-equal? (hash-ref run 'stopped) (cadr entry))
         (check-equal? (zero? (hash-ref run 'main_steps_per_s)) (eq? (car entry) 'idle)))
       (check-equal? (length runs) 4)))))
