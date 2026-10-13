#lang racket/base

(require (only-in racket/format ~r)
         (only-in racket/match match-define)
         (only-in threading ~>)
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
  (define (over-cap? made)
    (and (zero? (modulo made cap-check-every))
         (> (live) ledger-cap)
         (box-cas! capped #f #t)))
  (loop-for-seconds churn-seconds (lambda () (randn n)) #:stop? over-cap?))

(define (drain! p)
  (define draining
    (measure-span
     (lambda ()
       (let loop ([rounds 0])
         (define settled? (and (zero? ((prototype-live-bytes p))) (collect-past-canary!)))
         (cond
           [(or settled? (= rounds drain-round-limit)) (cons rounds settled?)]
           [else
            (collect-garbage 'major)
            (sleep 0.01)
            (loop (add1 rounds))])))))
  (match-define (cons rounds settled?) (span-value draining))
  (values rounds (span-ms draining) settled?))

(define (mib bytes)
  (~r (/ bytes 1048576.0) #:precision 1))

(define (churn-row p randn workers mode)
  (collect-garbage 'major)
  (define live (prototype-live-bytes p))
  (define peak (box #f))
  (define capped (box #f))
  (define churned
    (run-parallel/gc workers
                     (lambda (_i) (churn! randn live capped))
                     #:setup (lambda () (at-set-num-threads 1))
                     #:counter (prototype-finalizer-frees p)
                     #:main mode
                     #:on-wake (lambda () (set-box! peak (max (or (unbox peak) 0) (live))))))
  (define lag (live))
  (define ms (span-ms churned))
  (define interval (/ ms 1000.0))
  (define total-made (apply + (span-value churned)))
  (define frees (span-counted churned))
  (define-values (rounds drain-ms settled?) (drain! p))
  (values
   (list (prototype-name p)
         workers
         (main-mode-label mode)
         (fmt-rate (/ total-made interval))
         (fmt-rate (/ frees interval))
         (~r (/ frees (max total-made 1)) #:precision 2)
         (~> (span-gc-ms churned) (/ ms) (~r #:precision 2))
         (if (unbox peak) (~> (unbox peak) (max lag) mib) "not sampled")
         (if (unbox capped) (format "~a (cap)" (mib lag)) (mib lag))
         rounds
         (fmt-ms drain-ms)
         (live))
   settled?))

(module+ main
  (at-set-num-threads 1)
  (start-finalizer-thread!)
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
      (~> (list (if drained? "yes" "**NO: sweep stopped**") (load-average))
          (append row _)
          print-table-row)
      (and (not drained?)
           (format "~a with ~a workers, main ~a" (prototype-name p) workers (main-mode-label mode)))))
  (when undrained
    (printf "\nstopped: ~a did not drain in ~a major collections, ~a\n"
            undrained drain-round-limit
            "and its pending finalizers would run during every later row"))
  (print-footer))
