#lang racket/base

;; Which collection releases a step's dead temporaries (#145). K tensors
;; of 4 MiB live together, then drop as a batch; the ledger is read after
;; each step under one between-step policy.
;; Run:  POLICY=none|yield|minor|minor3|major|reclaim K=32 STEPS=16 \
;;       racket scripts/bench-gc-policy.rkt

(require (only-in torch/foreign
                  cpu-device finalizer-diagnostics native-memory-use
                  reclaim-native-memory! zeros))

(define (ledger-mib)
  (define e (assoc (cpu-device) (native-memory-use)))
  (quotient (if e (cdr e) 0) (* 1024 1024)))

(define (finalizer-runs)
  (cdr (assq 'runs (finalizer-diagnostics))))

(define minor (box 0))
(define major (box 0))
(define receiver (make-log-receiver (current-logger) 'debug 'GC))
(void
 (thread
  (lambda ()
    (let loop ()
      (define msg (vector-ref (sync receiver) 1))
      (cond
        [(regexp-match? #rx"MAJ" msg) (set-box! major (add1 (unbox major)))]
        [(regexp-match? #rx"min" msg) (set-box! minor (add1 (unbox minor)))]
        [else (void)])
      (loop)))))

(define K (string->number (or (getenv "K") "32")))
(define STEPS (string->number (or (getenv "STEPS") "16")))
(define POLICY (string->symbol (or (getenv "POLICY") "none")))

(define (step)
  (define held (for/list ([_ (in-range K)]) (zeros 1024 1024)))
  (length held))

(define (between)
  (case POLICY
    [(none) (void)]
    [(yield) (sleep 0)]
    [(minor) (collect-garbage 'minor) (sleep 0)]
    [(minor3) (for ([_ (in-range 3)]) (collect-garbage 'minor) (sleep 0))]
    [(major) (collect-garbage) (sleep 0)]
    [(reclaim) (reclaim-native-memory!)]
    [else (error 'bench-gc-policy "unknown POLICY: ~a" POLICY)]))

(module+ main
  (reclaim-native-memory!)
  (define base (ledger-mib))
  (printf "policy=~a K=~a (~a MiB/step) base=~a MiB\n" POLICY K (* 4 K) base)
  (printf "step ledger-MiB fin-runs minorGC majorGC gc-ms ms\n")
  (define t0 (current-inexact-milliseconds))
  (define gc0 (current-gc-milliseconds))
  (for ([i (in-range STEPS)])
    (step)
    (between)
    (sleep 0)
    (printf "~a ~a ~a ~a ~a ~a ~a\n"
            i
            (- (ledger-mib) base)
            (finalizer-runs)
            (unbox minor)
            (unbox major)
            (- (current-gc-milliseconds) gc0)
            (round (- (current-inexact-milliseconds) t0)))))
