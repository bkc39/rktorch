#lang racket/base

;; The #145 trough on a CPU training loop. `bench-gc-policy.rkt` measures
;; which collection releases a step's temporaries, but it allocates by hand and
;; never calls backward!, which is where the trough hook actually sits; this
;; drives a real forward/backward/step with adam so the hook is on the path.
;;
;;   MODE=pressure|off SCALE=small|large STEPS=40 racket scripts/bench-cpu-training.rkt
;;
;; The two sizes are the two regimes the shipped default produces on CPU. CPU
;; has no capacity to derive a mark from, so the backstop is off unless
;; native-memory-limit is set: the trough's margin is the whole policy here.
;;   small - a step's residue stays under the 256 MiB floor of the margin, so
;;           the policy should never fire and should cost nothing.
;;   large - residue clears the margin, so the trough should hold the ledger
;;           flat where MODE=off lets it climb.
;; RSS is a real gauge on this arm: CPU tensors are ordinary host allocations.

(require racket/format
         (only-in racket/os getpid)
         (only-in racket/port with-output-to-string)
         (only-in racket/string string-trim)
         (only-in racket/system system)
         torch
         torch/nn)

(define (env name default) (or (getenv name) default))
(define MODE (string->symbol (env "MODE" "pressure")))
(define SCALE (string->symbol (env "SCALE" "large")))
(define STEPS (string->number (env "STEPS" "40")))
(define EVERY (string->number (env "EVERY" "5")))

(define mib (* 1024 1024))
(define never (expt 2 60))

(define-values (WIDTH DEPTH BATCH)
  (case SCALE
    [(small) (values 32 3 16)]
    [(large) (values 96 4 48)]
    [else (error 'bench-cpu-training "SCALE is small or large, got ~a" SCALE)]))

(define (diagnostic key) (cdr (assq key (finalizer-diagnostics))))

(define (ledger-mib)
  (for/sum ([e (in-list (native-memory-use))]) (quotient (cdr e) mib)))

(define pid (number->string (getpid)))
(define (rss-mib)
  (define text
    (with-output-to-string
      (lambda () (void (system (string-append "ps -o rss= -p " pid))))))
  (define n (string->number (string-trim text)))
  (if n (quotient n 1024) 0))

(define majors (box 0))
(define receiver (make-log-receiver (current-logger) 'debug 'GC))
(void
 (thread
  (lambda ()
    (let loop ()
      (when (regexp-match? #rx"MAJ" (vector-ref (sync receiver) 1))
        (set-box! majors (add1 (unbox majors))))
      (loop)))))

(define (run)
  (manual-seed! 0)
  (define in-conv (Conv2d 3 WIDTH 3 #:padding 1))
  (define blocks
    (for/list ([_ (in-range DEPTH)])
      (cons (Conv2d WIDTH WIDTH 3 #:padding 1)
            (LayerNorm (list WIDTH 32 32)))))
  (define out-conv (Conv2d WIDTH 3 3 #:padding 1))
  (define params
    (append (parameters in-conv)
            (for*/list ([b (in-list blocks)]
                        [p (in-list (append (parameters (car b))
                                            (parameters (cdr b))))])
              p)
            (parameters out-conv)))
  (define opt (adam params #:lr 1e-4))
  (define (forward x)
    (out-conv
     (for/fold ([h (in-conv x)]) ([b (in-list blocks)])
       (relu ((cdr b) ((car b) h))))))
  (printf "mode=~a size=~a width=~a depth=~a batch=~a steps=~a\n"
          MODE SCALE WIDTH DEPTH BATCH STEPS)
  (displayln "step peak-ledger-MiB ledger-MiB entries racket-MiB rss-MiB troughs majors s/step loss")
  (define t0 (current-inexact-milliseconds))
  (for/fold ([window t0] [peak 0] #:result (void))
            ([i (in-range 1 (add1 STEPS))])
    (define x (randn BATCH 3 32 32))
    (define target (randn BATCH 3 32 32))
    (zero-grads! opt)
    (define loss (mse-loss (forward x) target))
    (backward! loss)
    (step! opt)
    (define l (item loss))
    (define seen (max peak (ledger-mib)))
    (cond
      [(zero? (remainder i EVERY))
       (define now (current-inexact-milliseconds))
       (printf "~a ~a ~a ~a ~a ~a ~a ~a ~a ~a\n"
               i seen (ledger-mib) (diagnostic 'ledger-entries)
               (quotient (current-memory-use) mib) (rss-mib)
               (diagnostic 'trough-collections) (unbox majors)
               (~r (/ (- now window) (* 1000.0 EVERY)) #:precision '(= 3))
               (~r l #:precision '(= 4)))
       (flush-output)
       (values now 0 )]
      [else (values window seen)]))
  (printf "total ~a s, gc ~a ms, troughs ~a, backstop ~a\n"
          (~r (/ (- (current-inexact-milliseconds) t0) 1000.0) #:precision '(= 1))
          (current-gc-milliseconds)
          (diagnostic 'trough-collections)
          (diagnostic 'pressure-collections)))

(module+ main
  (parameterize ([native-memory-limit never]  ;; CPU: backstop off either way
                 [native-collect-margin (and (eq? MODE 'off) never)]
                 [native-collect-budget (string->number (env "BUDGET" "1/20"))])
    (with-default-device 'cpu
      (run))
    (reclaim-native-memory!)
    (reclaim-native-memory!)
    (printf "after the model is dropped: ledger ~a MiB, entries ~a, racket ~a MiB, rss ~a MiB\n"
            (ledger-mib) (diagnostic 'ledger-entries)
            (quotient (current-memory-use) mib) (rss-mib))))
