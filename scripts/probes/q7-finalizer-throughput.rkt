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

(define (churn! randn live capped stop t i)
  (define (mark-if-over-cap! made)
    (when (and (zero? (modulo made cap-check-every)) (> (live) ledger-cap))
      (set-box! capped #t)))
  (tally-loop-for-seconds t i churn-seconds
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
  (define t (tally workers))
  (define churned
    (run-parallel workers
                  (lambda (i) (churn! randn live capped stop t i))
                  #:setup (lambda () (at-set-num-threads 1))
                  #:counters (list (lambda () (tally-total t)) (prototype-finalizer-frees p))
                  #:main mode
                  #:on-start (lambda () (tally-open! t))
                  #:on-end (lambda () (tally-close! t))
                  #:stop stop
                  #:on-wake (lambda () (set-box! peak (max (or (unbox peak) 0) (live))))))
  (define lag (live))
  (define lag-mib (bytes->mib-string lag))
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
         (gap->string (tally-max-gap t))
         (if (unbox peak) (~> (unbox peak) (max lag) bytes->mib-string) "not sampled")
         (if (unbox capped) (format "~a (cap)" lag-mib) lag-mib)
         collections
         (real->decimal-string drain-ms 1)
         (live))
   collections
   settled?))

(module+ main
  (at-set-num-threads 1)
  (collect-past-canary!)
  (print-banner
   (format "Q7: finalizer throughput, ~a x ~a randn for ~a s or ~a MiB"
           n n churn-seconds (bytes->mib-string ledger-cap))
   #:torch-version (shim-version))
  (print-table-header
   '("allocator" "workers" "main thread" "made/s" "frees/s" "frees / made" "GC share"
     "longest gap ms" "peak ledger MiB" "ledger MiB at stop" "major collections to drain"
     "drain ms" "ledger bytes after drain" "drained" "load"))
  (define undrained
    (with-handlers ([exn:fail:unsettled? exn-message])
      (for*/or ([make (in-list (variant-makers))]
                [workers (in-list worker-counts)]
                [mode (in-list modes)])
        (match-define (variant p o) (make))
        (define-values (row collections drained?) (churn-row p (ops-randn o) workers mode))
        (~> (list (if drained? "yes" "NO") (load-average))
            (append row _)
            print-table-row)
        (and (not drained?)
             (format "~a, ~a workers, main ~a: not drained after ~a major collections"
                     (prototype-name p) workers (main-mode-label mode) collections)))))
  (when undrained
    (printf "\nstopped: ~a\n" undrained))
  (print-footer))
