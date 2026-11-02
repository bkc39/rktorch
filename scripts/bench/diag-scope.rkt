#lang racket/base

;; Diagnostic for tensor scopes on the loader path: racket diag-scope.rkt
;; [--seconds s] [--warmup s] [--reps n] [--workers 0,1,2] [--modes consume,train]
;; [--arms off,build,ref] runs every cell in a fresh process and prints one
;; line per run; --cell ARM WORKERS MODE runs one cell and writes its record.

(require (only-in racket/cmdline command-line)
         (only-in racket/math exact-round)
         (only-in racket/port with-output-to-string)
         (only-in racket/string string-split)
         (only-in racket/system process*/ports)
         (only-in torch
                  * / @ - copy! flip narrow rand reshape select
                  shape sum tensor to-dtype zeros ~>)
         (only-in torch/data/loader dataloader define-dataset)
         (only-in torch/foreign/raw/collector finalizer-runs finalizer-timing)
         (only-in torch/foreign/raw/scope scope-diagnostics)
         (only-in "diag-counters.rkt"
                  clock-ms ledger-entries load-average median-record peak-rss-kb))

(define samples 4096)
(define batch-size 64)
(define main-steps-per-batch 8)

(define-dataset cifarish (xs ys mean std)
  #:length (car (shape xs))
  #:ref (i)
  (define pixels (~> (select xs 0 i) (to-dtype 'float32) (/ 255.0)))
  (define flipped (if (< (random) 0.5) (flip pixels 2) pixels))
  (define canvas (zeros 3 40 40))
  (copy! (~> canvas (narrow 1 4 32) (narrow 2 4 32)) flipped)
  (values (~> canvas (narrow 1 (random 9) 32) (narrow 2 (random 9) 32) (- mean) (/ std))
          (select ys 0 i)))

(define (run-cell workers mode seconds warmup)
  (define xs (to-dtype (* (rand samples 3 32 32) 255.0) 'uint8))
  (define ys (to-dtype (* (rand samples) 10.0) 'int64))
  (define mean (reshape (tensor '(0.49 0.48 0.45)) 3 1 1))
  (define std (reshape (tensor '(0.25 0.24 0.26)) 3 1 1))
  (define loader
    (if (zero? workers)
        (dataloader (cifarish xs ys mean std) #:batch-size batch-size #:shuffle? #t)
        (dataloader (cifarish xs ys mean std) #:batch-size batch-size #:shuffle? #t
                    #:workers workers)))
  (define w (rand 3072 10))
  (define (consume! x)
    (case mode
      [(consume) (sum x) 0]
      [(train)
       (define flat (reshape x (car (shape x)) 3072))
       (for ([_ (in-range main-steps-per-batch)]) (sum (@ flat w)))
       main-steps-per-batch]))
  (define (epochs-until deadline)
    (let loop ([n 0] [steps 0])
      (cond
        [(>= (clock-ms) deadline) (values n steps)]
        [else
         (define-values (k s)
           (for/fold ([k 0] [s 0]) ([vs (in-values-sequence loader)])
             (values (+ k (car (shape (car vs)))) (+ s (consume! (car vs))))))
         (loop (+ n k) (+ steps s))])))
  ;; free: main steps flat out while a green thread drains the loader
  (define consumed (box 0))
  (define drainer
    (and (eq? mode 'free)
         (thread (lambda ()
                   (let loop ()
                     (for ([vs (in-values-sequence loader)])
                       (sum (car vs))
                       (set-box! consumed (+ (unbox consumed) (car (shape (car vs))))))
                     (loop))))))
  (define own (rand batch-size 3072))
  (define (steps-until deadline)
    (define c0 (unbox consumed))
    (let loop ([steps 0])
      (cond
        [(>= (clock-ms) deadline) (values (- (unbox consumed) c0) steps)]
        [else (sum (@ own w)) (sleep 0) (loop (add1 steps))])))
  (define run-until (if drainer steps-until epochs-until))
  (run-until (+ (clock-ms) (* 1000 warmup)))
  (define peak (box (ledger-entries)))
  (define sampler
    (thread (lambda ()
              (let loop ()
                (set-box! peak (max (unbox peak) (ledger-entries)))
                (sleep 0.02)
                (loop)))))
  (define runs0 (finalizer-runs))
  (define-values (main0 us0) (finalizer-timing))
  (define scope0 (scope-diagnostics))
  (define gc0 (current-gc-milliseconds))
  (define t0 (clock-ms))
  (define-values (n steps) (run-until (+ t0 (* 1000 seconds))))
  (define elapsed (/ (- (clock-ms) t0) 1000.0))
  (define gc (- (current-gc-milliseconds) gc0))
  (define runs (- (finalizer-runs) runs0))
  (define-values (main1 us1) (finalizer-timing))
  (define scope1 (scope-diagnostics))
  (kill-thread sampler)
  (when drainer (kill-thread drainer))
  (define (per-s v) (/ v elapsed))
  (append
   (list (cons 'samples/s (per-s n))
         (cons 'steps/s (per-s steps))
         (cons 'finalizer-runs/s (per-s runs))
         (cons 'finalizer-main-runs/s (per-s (- main1 main0)))
         (cons 'finalizer-ms/s (per-s (/ (- us1 us0) 1000.0)))
         (cons 'gc-ms/s (per-s gc))
         (cons 'peak-ledger-entries (unbox peak))
         (cons 'peak-rss-mb (let ([kb (peak-rss-kb)]) (and kb (quotient kb 1024))))
         (cons 'elapsed-s elapsed))
   (for/list ([(k v) (in-hash scope1)])
     (cons (string->symbol (format "~a/s" k)) (per-s (- v (hash-ref scope0 k)))))))

(define (run-child arm workers mode seconds warmup)
  (define env (environment-variables-copy (current-environment-variables)))
  (environment-variables-set! env #"RKTORCH_DIAG_SCOPE" (string->bytes/utf-8 (symbol->string arm)))
  (environment-variables-set! env #"OMP_NUM_THREADS" #"1")
  (define out
    (parameterize ([current-environment-variables env])
      (with-output-to-string
        (lambda ()
          (define p
            (list-ref
             (process*/ports (current-output-port) #f (current-error-port)
                            (find-executable-path (find-system-path 'exec-file))
                            (path->string (syntax-source #'here))
                            "--seconds" (number->string seconds)
                            "--warmup" (number->string warmup)
                            "--cell" (symbol->string arm) (number->string workers)
                            (symbol->string mode))
             4))
          (p 'wait)))))
  (read (open-input-string out)))

(define (show v)
  (cond
    [(not v) "-"]
    [(and (real? v) (>= (abs v) 100)) (number->string (exact-round v))]
    [(real? v) (real->decimal-string v 1)]
    [else (format "~a" v)]))

(define (drive arms workers-list modes reps seconds warmup)
  (for* ([mode (in-list modes)] [workers (in-list workers-list)])
    (printf "# mode ~a workers ~a load ~a\n" mode workers (load-average))
    (define runs
      (for*/list ([rep (in-range reps)] [arm (in-list arms)])
        (define r (run-child arm workers mode seconds warmup))
        (printf "~s\n" (list 'run mode workers arm rep r))
        (flush-output)
        (cons arm r)))
    (for ([arm (in-list arms)])
      (define m (median-record (for/list ([r (in-list runs)] #:when (eq? (car r) arm)) (cdr r))))
      (printf "~s\n" (list 'median mode workers arm m))
      (printf "| ~a | ~a | ~a | ~a |\n" mode workers arm
              (apply string-append
                     (for/list ([kv (in-list m)])
                       (format "~a=~a " (car kv) (show (cdr kv))))))
      (flush-output))))

(module+ main
  (define seconds 10)
  (define warmup 2)
  (define reps 3)
  (define arms '(off build ref))
  (define workers-list '(0 1 2 4 8))
  (define modes '(consume train))
  (define (symbols s) (map string->symbol (string-split s ",")))
  (define cell #f)
  (command-line
     #:once-each
     [("--seconds") s "Timed seconds per run" (set! seconds (string->number s))]
     [("--warmup") s "Warm-up seconds per run" (set! warmup (string->number s))]
     [("--reps") n "Repetitions per cell" (set! reps (string->number n))]
     [("--arms") a "Comma-separated arms" (set! arms (symbols a))]
     [("--workers") w "Comma-separated worker counts"
                    (set! workers-list (map string->number (string-split w ",")))]
     [("--modes") m "Comma-separated main modes" (set! modes (symbols m))]
   [("--cell") arm workers mode "Run one cell in this process"
               (set! cell (list (string->symbol arm) (string->number workers)
                                (string->symbol mode)))])
  (cond
    [cell
     (write (run-cell (cadr cell) (caddr cell) seconds warmup))
     (newline)]
    [else (drive arms workers-list modes reps seconds warmup)]))
