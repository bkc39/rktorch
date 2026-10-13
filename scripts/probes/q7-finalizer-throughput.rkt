#lang racket/base

(require (only-in racket/format ~r)
         (only-in racket/match match-define)
         "common.rkt"
         "prototype-allocator.rkt"
         "shim.rkt"
         "variants.rkt")

(define churn-seconds 2.0)
(define n 64)
(define worker-counts '(1 4 8))
(define drain-round-limit 200)

(define (churn! randn)
  (define deadline (+ (current-inexact-monotonic-milliseconds) (* 1000 churn-seconds)))
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
    (define settled? (and (zero? ((prototype-live-bytes p))) (collect-past-canary!)))
    (cond
      [(or settled? (= rounds drain-round-limit))
       (values rounds (- (current-inexact-monotonic-milliseconds) start) settled?)]
      [else
       (collect-garbage 'major)
       (sleep 0.01)
       (loop (add1 rounds))])))

(define (churn-row p randn workers)
  (collect-garbage 'major)
  (define finished (box #f))
  (define failure (box #f))
  (define at-stop (box #f))
  (define churner
    (thread (lambda ()
              (with-handlers ([(lambda (_) #t) (lambda (e) (set-box! failure e))])
                (define-values (ms made gc-ms runs)
                  (run-parallel/gc workers
                                   (lambda (_i) (churn! randn))
                                   #:setup (lambda () (at-set-num-threads 1))
                                   #:counter (prototype-finalizer-runs p)))
                (set-box! at-stop
                          (vector (apply + made)
                                  (/ ms 1000.0)
                                  runs
                                  gc-ms
                                  ((prototype-live-bytes p)))))
              (set-box! finished #t))))
  (define peak (sample-until (lambda () (unbox finished)) p))
  (thread-wait churner)
  (when (unbox failure) (raise (unbox failure)))
  (define-values (made interval runs-during gc-ms lag)
    (vector->values (unbox at-stop)))
  (define gc-share (/ gc-ms (* 1000 interval)))
  (define-values (rounds drain-ms settled?) (drain! p))
  (define after ((prototype-live-bytes p)))
  (values
   (list (prototype-name p)
         workers
         (fmt-rate (/ made interval))
         (fmt-rate (/ runs-during interval))
         (~r (/ runs-during (max made 1)) #:precision 2)
         (~r gc-share #:precision 2)
         (~r (/ (max peak lag) 1048576.0) #:precision 1)
         (~r (/ lag 1048576.0) #:precision 1)
         rounds
         (fmt-ms drain-ms)
         after)
   settled?))


(module+ main
  (at-set-num-threads 1)
  ;; Registering first from one thread starts exactly one finalizer thread,
  ;; so the drain's canary runs after every earlier callback has returned.
  (void (collect-past-canary!))
  (print-banner
   (format "Q7: finalizer throughput, ~a x ~a float32 randn made and dropped for ~a s"
           n n churn-seconds)
   #:torch-version (shim-version))
  (print-table-header
   '("allocator" "workers" "tensors made/s" "finalizer runs/s during churn"
     "runs / made" "GC ms per wall ms" "peak ledger MiB" "ledger MiB at stop"
     "major collections to drain" "drain ms" "ledger bytes after drain" "drained" "load"))
  (define undrained
    (for*/or ([make (in-list (variant-makers))]
              [workers (in-list worker-counts)])
      (match-define (variant p o) (make))
      (define-values (row drained?) (churn-row p (ops-randn o) workers))
      (print-table-row
       (append row (list (if drained? "yes" "**NO: sweep stopped**") (load-average))))
      (and (not drained?) (format "~a with ~a workers" (prototype-name p) workers))))
  (when undrained
    (printf "\nstopped: ~a did not drain in ~a major collections, ~a\n"
            undrained drain-round-limit
            "and its pending finalizers would run during every later row"))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
