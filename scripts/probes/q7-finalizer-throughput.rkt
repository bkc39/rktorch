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
(define ledger-cap (mib->bytes 512))
(define cap-check-every 256)
(define modes (append (main-modes churn-seconds) (list (main-polling 50))))

(define (churn! randn live capped)
  (define (mark-if-over-cap! made)
    (when (and (zero? (modulo made cap-check-every)) (> (live) ledger-cap))
      (set-box! capped #t)))
  (loop-for-seconds churn-seconds
                    (lambda (made)
                      (randn n)
                      (mark-if-over-cap! made))
                    #:stop? (lambda (_made) (unbox capped))))

(define (drain! p)
  (define draining
    (measure-span
     (lambda ()
       (let loop ([rounds 0])
         (define settled? (and (zero? ((prototype-live-bytes p))) (settle-wills!) #t))
         (cond
           [(or settled? (= rounds drain-round-limit)) (cons rounds settled?)]
           [else
            (collect-garbage 'major)
            (sleep 0.01)
            (loop (add1 rounds))])))))
  (match-define (cons rounds settled?) (span-value draining))
  (values rounds (span-ms draining) settled?))

(define (bytes->mib-string bytes)
  (mib->string (bytes->mib bytes)))

(define (churn-row p randn workers mode)
  (collect-garbage 'major)
  (define live (prototype-live-bytes p))
  (define peak (box #f))
  (define capped (box #f))
  (define churned
    (run-parallel workers
                  (lambda (_i) (churn! randn live capped))
                  #:setup (lambda () (at-set-num-threads 1))
                  #:counter (prototype-finalizer-frees p)
                  #:main mode
                  #:on-wake (lambda () (set-box! peak (max (or (unbox peak) 0) (live))))))
  (define lag (live))
  (define ms (span-ms churned))
  (define interval (ms->seconds ms))
  (define total-made (apply + (span-value churned)))
  (define frees (span-counted churned))
  (define-values (rounds drain-ms settled?) (drain! p))
  (values
   (list (prototype-name p)
         workers
         (main-mode-label mode)
         (rate->string (/ total-made interval))
         (rate->string (/ frees interval))
         (~r (/ frees (max total-made 1)) #:precision 2)
         (~> (span-gc-ms churned) (/ ms) (~r #:precision 2))
         (if (unbox peak) (~> (unbox peak) (max lag) bytes->mib-string) "not sampled")
         (if (unbox capped) (format "~a (cap)" (bytes->mib-string lag)) (bytes->mib-string lag))
         rounds
         (ms->string drain-ms)
         (live))
   settled?))

(module+ main
  (at-set-num-threads 1)
  (unless (collect-past-canary!)
    (error 'q7 "the finalizer thread did not run a canary"))
  (print-banner
   (format "Q7: finalizer throughput, ~a x ~a float32 randn made and dropped by worker threads for ~a s or until the ledger passes ~a MiB, under each main-thread mode; frees counted during the churn, then drained by major collections"
           n n churn-seconds (bytes->mib-string ledger-cap))
   #:torch-version (shim-version))
  (print-table-header
   '("allocator" "workers" "main thread while workers churn" "tensors made/s"
     "finalizer frees/s during churn" "frees / made" "GC wall ms per wall ms (excludes rendezvous wait)"
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
    (printf "\nstopped: ~a did not drain in ~a major collections, and its pending finalizers would run during every later row\n"
            undrained drain-round-limit))
  (print-footer))
