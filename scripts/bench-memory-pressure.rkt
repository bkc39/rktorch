#lang racket/base

;; Native memory over a training loop (#145): a stack of Conv2d + LayerNorm
;; + relu blocks on random [BATCH 3 32 32] batches with adam, reporting the
;; caching allocator's peak per ten-step window, reserved bytes, seconds per
;; step and the ledger's pressure collections.
;; Run:  MODE=off|manual|backstop|pressure NOGRAD=1 BATCH=256 WIDTH=128 \
;;       DEPTH=8 STEPS=60 \
;;       GC_EVERY=10 LIMIT=<MiB> BUDGET=0.05 racket scripts/bench-memory-pressure.rkt
;; off disables every collection the ledger makes, manual collects by hand
;; every GC_EVERY steps on top of that, backstop keeps only the mid-forward
;; trigger, pressure is the library as shipped (LIMIT overrides the
;; capacity-derived mark). NOGRAD=1 runs the forward alone under with-no-grad,
;; the shape of a sampling or evaluation loop. Each row also carries what a
;; leak would move: ledger entries, Racket's heap and the process RSS; the
;; last line is the ledger and the allocator after the model is dropped.

(require racket/format
         (only-in racket/string string-split)
         torch
         (only-in torch/foreign/raw/memory trough-budget trough-margin)
         torch/nn)

(define (env name default)
  (or (getenv name) default))

(define MODE (string->symbol (env "MODE" "pressure")))
(define BATCH (string->number (env "BATCH" "256")))
(define WIDTH (string->number (env "WIDTH" "128")))
(define DEPTH (string->number (env "DEPTH" "8")))
(define STEPS (string->number (env "STEPS" "60")))
(define NOGRAD? (and (getenv "NOGRAD") #t))
(define GC-EVERY (string->number (env "GC_EVERY" "10")))
(define LIMIT
  (let ([mib (getenv "LIMIT")])
    (and mib (* 1024 1024 (string->number mib)))))

(define TRACE
  (let ([from (getenv "TRACE")])
    (and from (string->number from))))

(define major-collections (box 0))
(define gc-receiver (make-log-receiver (current-logger) 'debug 'GC))
(void
 (thread
  (lambda ()
    (let loop ()
      (when (regexp-match? #rx"MAJ" (vector-ref (sync gc-receiver) 1))
        (set-box! major-collections (add1 (unbox major-collections))))
      (loop)))))

(define never (expt 2 60))
(define mib (* 1024 1024))

(define (stat key)
  (quotient (cdr (assq key (cuda-memory-stats))) mib))

(define (diagnostic key)
  (cdr (assq key (finalizer-diagnostics))))

(define (rss-mib)
  (if (file-exists? "/proc/self/status")
      (call-with-input-file "/proc/self/status"
        (lambda (in)
          (for/first ([l (in-lines in)]
                      #:when (regexp-match? #rx"^VmRSS" l))
            (quotient (string->number (cadr (string-split l))) 1024))))
      0))

(define (report-state label)
  (printf "~a: ledger ~a MiB, allocated ~a MiB, reserved ~a MiB, ~s\n"
          label
          (for/sum ([entry (in-list (native-memory-use))])
            (quotient (cdr entry) mib))
          (stat 'allocated)
          (stat 'reserved)
          (finalizer-diagnostics)))

(define (make-blocks)
  (for/list ([_ (in-range DEPTH)])
    (cons (Conv2d WIDTH WIDTH 3 #:padding 1)
          (LayerNorm (list WIDTH 32 32)))))

(define (run)
  (define in-conv (Conv2d 3 WIDTH 3 #:padding 1))
  (define blocks (make-blocks))
  (define out-conv (Conv2d WIDTH 3 3 #:padding 1))
  (define params
    (append (parameters in-conv)
            (for*/list ([b (in-list blocks)]
                        [p (in-list (append (parameters (car b))
                                            (parameters (cdr b))))])
              p)
            (parameters out-conv)))
  (define opt (adam params #:lr 0.0001))
  (define (forward x)
    (out-conv
     (for/fold ([h (in-conv x)]) ([b (in-list blocks)])
       (relu ((cdr b) ((car b) h))))))
  (define model (procedure->Layer forward))
  (define (train-step!)
    (define x (randn BATCH 3 32 32))
    (cond
      [NOGRAD? (item (mean (with-no-grad (model x))))]
      [else
       (define target (randn BATCH 3 32 32))
       (zero-grads! opt)
       (define loss (mse-loss (forward x) target))
       (backward! loss)
       (step! opt)
       (item loss)]))
  (printf "mode=~a batch=~a width=~a depth=~a limit=~a total=~a MiB\n"
          MODE BATCH WIDTH DEPTH (getenv "LIMIT")
          (quotient (cdr (assq 'total (cuda-memory-info))) mib))
  (displayln "step window-peak-MiB allocated-MiB reserved-MiB backstop trough minors entries racket-MiB rss-MiB s/step")
  (cuda-reset-peak-stats!)
  (define gc0 (current-gc-milliseconds))
  (define t0 (current-inexact-milliseconds))
  (for/fold ([window-start t0]) ([i (in-range 1 (add1 STEPS))])
    (train-step!)
    (when (and TRACE (>= i TRACE))
      (printf "  ~a: ledger ~a allocated ~a peak ~a pressure ~a majors ~a racket ~a MiB\n"
              i
              (for/sum ([entry (in-list (native-memory-use))])
                (quotient (cdr entry) mib))
              (stat 'allocated)
              (stat 'peak-allocated)
              (diagnostic 'pressure-collections)
              (unbox major-collections)
              (quotient (current-memory-use) mib)))
    (when (and (eq? MODE 'manual) (zero? (remainder i GC-EVERY)))
      (collect-garbage)
      (sleep 0))
    (cond
      [(zero? (remainder i 10))
       (define now (current-inexact-milliseconds))
       (printf "~a ~a ~a ~a ~a ~a ~a ~a ~a ~a ~a\n"
               i
               (stat 'peak-allocated)
               (stat 'allocated)
               (stat 'reserved)
               (diagnostic 'pressure-collections)
               (diagnostic 'trough-collections)
               (diagnostic 'trough-minors)
               (diagnostic 'ledger-entries)
               (quotient (current-memory-use) mib)
               (rss-mib)
               (~r (/ (- now window-start) 10000.0) #:precision '(= 3)))
       (flush-output)
       (cuda-reset-peak-stats!)
       now]
      [else window-start]))
  (printf "total ~a s, gc ~a ms, reclaimed ~a MiB: ~a backstop, ~a trough, ~a minors\n"
          (~r (/ (- (current-inexact-milliseconds) t0) 1000.0) #:precision '(= 1))
          (- (current-gc-milliseconds) gc0)
          (quotient (diagnostic 'pressure-reclaimed) mib)
          (diagnostic 'pressure-collections)
          (diagnostic 'trough-collections)
          (diagnostic 'trough-minors)))

(module+ main
  (unless (cuda-available?)
    (error 'bench-memory-pressure "needs a CUDA device"))
  (manual-seed! 0)
  (parameterize ([native-memory-limit (if (memq MODE '(pressure backstop))
                                          LIMIT
                                          never)]
                 [trough-margin (if (eq? MODE 'pressure) #f never)]
                 [trough-budget (string->number (env "BUDGET" "1/20"))])
    (with-default-device (cuda-device)
      (with-handlers ([exn:fail:rktorch:oom?
                       (lambda (e)
                         (printf "OOM: ~a\n" (car (regexp-split #rx"\n" (exn-message e))))
                         (report-state "at the failure")
                         (reclaim-native-memory!)
                         (report-state "after reclaim")
                         (exit 1))])
        (run))
      (reclaim-native-memory!)
      (reclaim-native-memory!)
      (report-state "after the model is dropped"))))
