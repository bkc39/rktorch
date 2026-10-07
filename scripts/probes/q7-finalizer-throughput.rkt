#lang racket/base

;; #262 question 7: N parallel workers make tensors and drop them to the
;; collector; how fast do the finalizers run, and how far does the ledger
;; lag behind what is really live?

(require (only-in racket/format ~r)
         "common.rkt"
         "prototype-allocator.rkt"
         "shim.rkt"
         "variants.rkt")

(define churn-seconds 2.0)
(define n 64)
(define worker-counts '(1 4 8))

(define (churn! randn deadline)
  (let loop ([made 0])
    (cond
      [(> (current-inexact-monotonic-milliseconds) deadline) made]
      [else (randn n) (loop (add1 made))])))

(define (sample-until done? p)
  (let loop ([peak 0])
    (cond
      [(done?) peak]
      [else
       (sleep 0.05)
       (loop (max peak ((prototype-live-bytes p))))])))

(define (drain! p)
  (define start (current-inexact-monotonic-milliseconds))
  (let loop ([rounds 0])
    (cond
      [(or (zero? ((prototype-live-bytes p))) (= rounds 200))
       (values rounds (- (current-inexact-monotonic-milliseconds) start))]
      [else
       (collect-garbage 'major)
       (sleep 0.01)
       (loop (add1 rounds))])))

(define (churn-row p randn workers)
  (collect-garbage 'major)
  (define runs-before ((prototype-finalizer-runs p)))
  (define gc-before (current-gc-milliseconds))
  (define deadline (+ (current-inexact-monotonic-milliseconds) (* 1000 churn-seconds)))
  (define finished (box #f))
  (define made-box (box 0))
  (define interval-box (box #f))
  (define churner
    (thread (lambda ()
              (define-values (ms made)
                (run-parallel workers
                              (lambda (_i)
                                (at-set-num-threads 1)
                                (churn! randn deadline))))
              (set-box! made-box (apply + made))
              (set-box! interval-box (/ ms 1000.0))
              (set-box! finished #t))))
  (define peak (sample-until (lambda () (unbox finished)) p))
  (thread-wait churner)
  (define made (unbox made-box))
  (define runs-during (- ((prototype-finalizer-runs p)) runs-before))
  (define interval (unbox interval-box))
  (define gc-share (/ (- (current-gc-milliseconds) gc-before) (* 1000 interval)))
  (define lag ((prototype-live-bytes p)))
  (define-values (rounds drain-ms) (drain! p))
  (list (prototype-name p)
        workers
        (fmt-rate (/ made interval))
        (fmt-rate (/ runs-during interval))
        (~r (/ runs-during (max made 1)) #:precision 2)
        (~r gc-share #:precision 2)
        (~r (/ peak 1048576.0) #:precision 1)
        (~r (/ lag 1048576.0) #:precision 1)
        rounds
        (fmt-ms drain-ms)
        ((prototype-live-bytes p))
        (load-average)))

(module+ main
  (at-set-num-threads 1)
  (print-banner
   (format "Q7: finalizer throughput, ~a x ~a float32 randn made and dropped for ~a s"
           n n churn-seconds)
   #:torch-version (shim-version))
  (print-table-header
   '("allocator" "workers" "tensors made/s" "finalizer runs/s during churn"
     "runs / made" "GC share of wall" "peak ledger MiB" "ledger MiB at stop"
     "major collections to drain" "drain ms" "ledger bytes after drain" "load"))
  (for* ([v (in-list (workload-variants))]
         [workers (in-list worker-counts)])
    (define p (car v))
    (print-table-row (churn-row p (ops-randn (cdr v)) workers)))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
