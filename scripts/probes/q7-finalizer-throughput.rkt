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
(define ledger-cap (* 512 1024 1024))
(define cap-check-every 256)
(define modes (list main-blocked (main-polling 10) (main-polling 50) main-busy))

(define (churn! randn live capped)
  (define deadline (+ (current-inexact-monotonic-milliseconds) (* 1000 churn-seconds)))
  (let loop ([made 0])
    (cond
      [(> (current-inexact-monotonic-milliseconds) deadline) made]
      [(and (zero? (modulo made cap-check-every)) (> (live) ledger-cap))
       (set-box! capped #t)
       made]
      [else (randn n) (loop (add1 made))])))

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

(define (mib bytes)
  (~r (/ bytes 1048576.0) #:precision 1))

(define (churn-row p randn workers mode)
  (collect-garbage 'major)
  (define live (prototype-live-bytes p))
  (define peak (box #f))
  (define capped (box #f))
  (define-values (ms made gc-ms frees)
    (run-parallel/gc workers
                     (lambda (_i) (churn! randn live capped))
                     #:setup (lambda () (at-set-num-threads 1))
                     #:counter (prototype-finalizer-frees p)
                     #:main mode
                     #:on-wake (lambda () (set-box! peak (max (or (unbox peak) 0) (live))))))
  (define lag (live))
  (define interval (/ ms 1000.0))
  (define total-made (apply + made))
  (define-values (rounds drain-ms settled?) (drain! p))
  (values
   (list (prototype-name p)
         workers
         (main-mode-label mode)
         (fmt-rate (/ total-made interval))
         (fmt-rate (/ frees interval))
         (~r (/ frees (max total-made 1)) #:precision 2)
         (~r (/ gc-ms ms) #:precision 2)
         (if (unbox peak) (mib (max (unbox peak) lag)) "not sampled")
         (if (unbox capped) (format "~a (cap)" (mib lag)) (mib lag))
         rounds
         (fmt-ms drain-ms)
         (live))
   settled?))

(module+ main
  (at-set-num-threads 1)
  ;; Registering first from one thread starts exactly one finalizer thread,
  ;; so the drain's canary runs after every earlier callback has returned.
  (void (collect-past-canary!))
  (print-banner
   (format "Q7: finalizer throughput, ~a x ~a float32 randn made and dropped for ~a s or until the ledger passes ~a MiB; finalizers run only when the main thread's scheduler wakes"
           n n churn-seconds (mib ledger-cap))
   #:torch-version (shim-version))
  (print-table-header
   '("allocator" "workers" "main thread while workers churn" "tensors made/s"
     "finalizer frees/s during churn" "frees / made" "GC ms per wall ms"
     "peak ledger MiB, sampled when main wakes" "ledger MiB at stop"
     "major collections to drain" "drain ms" "ledger bytes after drain" "drained" "load"))
  (define undrained
    (for*/or ([make (in-list (variant-makers))]
              [workers (in-list worker-counts)]
              [mode (in-list modes)])
      (match-define (variant p o) (make))
      (define-values (row drained?) (churn-row p (ops-randn o) workers mode))
      (print-table-row
       (append row (list (if drained? "yes" "**NO: sweep stopped**") (load-average))))
      (and (not drained?)
           (format "~a with ~a workers, main ~a" (prototype-name p) workers (main-mode-label mode)))))
  (when undrained
    (printf "\nstopped: ~a did not drain in ~a major collections, ~a\n"
            undrained drain-round-limit
            "and its pending finalizers would run during every later row"))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
