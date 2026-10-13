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

(define (churn! randn live capped stop tally i)
  (define (mark-if-over-cap! made)
    (when (and (zero? (modulo made cap-check-every)) (> (live) ledger-cap))
      (set-box! capped #t)))
  (tally-loop-for-seconds tally i churn-seconds
                          (lambda (made)
                            (randn n)
                            (mark-if-over-cap! made))
                          #:stop? (lambda (_made) (or (unbox capped) (unbox stop)))))

(define (drain! p)
  (define live (prototype-live-bytes p))
  (define draining
    (measure-span
     (lambda ()
       (let loop ([rounds 0] [collections 0])
         (define-values (ran settle-collections)
           (cond
             [(zero? (live)) (settle-wills!)]
             [else (values #f 0)]))
         (define so-far (+ collections settle-collections))
         (cond
           [ran (cons so-far #t)]
           [(= rounds drain-round-limit) (cons so-far #f)]
           [else
            (collect-garbage 'major)
            (sleep 0.01)
            (loop (add1 rounds) (add1 so-far))])))))
  (match-define (cons collections settled?) (span-value draining))
  (values collections (span-ms draining) settled?))

(define (churn-row p randn workers mode)
  (collect-garbage 'major)
  (define live (prototype-live-bytes p))
  (define peak (box #f))
  (define capped (box #f))
  (define stop (box #f))
  (define tally (make-tally workers))
  (define churned
    (run-parallel workers
                  (lambda (i) (churn! randn live capped stop tally i))
                  #:setup (lambda () (at-set-num-threads 1))
                  #:counters (list (lambda () (tally-total tally)) (prototype-finalizer-frees p))
                  #:main mode
                  #:stop stop
                  #:on-wake (lambda () (set-box! peak (max (or (unbox peak) 0) (live))))))
  (define lag (live))
  (define ms (span-ms churned))
  (define interval (ms->seconds ms))
  (match-define (list made frees) (span-counts churned))
  (define-values (collections drain-ms settled?) (drain! p))
  (values
   (list (prototype-name p)
         workers
         (main-mode-label mode)
         (rate->string (/ made interval))
         (rate->string (/ frees interval))
         (~r (/ frees (max made 1)) #:precision 2)
         (~> (span-gc-ms churned) (/ ms) (~r #:precision 2))
         (real->decimal-string (tally-max-gap tally) 1)
         (if (unbox peak) (~> (unbox peak) (max lag) bytes->mib-string) "not sampled")
         (if (unbox capped) (format "~a (cap)" (bytes->mib-string lag)) (bytes->mib-string lag))
         collections
         (real->decimal-string drain-ms 1)
         (live))
   settled?))

(module+ main
  (at-set-num-threads 1)
  (collect-past-canary!)
  (print-banner
   (format "Q7: finalizer throughput, ~a x ~a float32 randn made and dropped by worker threads for ~a s or until the ledger passes ~a MiB, under each main-thread mode; made and frees counted in the window (until the workers exit, or in the foreign-call modes until main's last wake), then drained by major collections"
           n n churn-seconds (bytes->mib-string ledger-cap))
   #:torch-version (shim-version))
  (print-table-header
   '("allocator" "workers" "main thread while workers churn" "tensors made/s"
     "finalizer frees/s in the window" "frees / made" "GC wall ms per wall ms (excludes rendezvous wait)"
     "longest ms between a worker's iterations"
     "peak ledger MiB, sampled when main wakes" "ledger MiB at stop"
     "major collections to drain, settling's included" "drain ms" "ledger bytes after drain" "drained" "load"))
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
