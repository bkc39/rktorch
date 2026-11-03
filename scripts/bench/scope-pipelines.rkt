#lang racket/base

;; Loader throughput of three pipelines (workloads/pipelines.rkt) under the
;; tensor-scope arms against their PyTorch twins (python/pipelines.py):
;;   racket scope-pipelines.rkt --out runs.jsonl [--pipelines jpeg,cifar,text]
;;     [--workers 0,2,4,8] [--arms off,build,ref] [--reps 3]
;;     [WARMUP_EPOCHS=2 TIMED_EPOCHS=3]
;; runs every cell in a fresh process, PyTorch interleaved with the arms, and
;; appends one JSON line per run; --summarize runs.jsonl prints the median
;; tables; --cell PIPELINE WORKERS runs one rktorch cell under the arm in
;; RKTORCH_DIAG_SCOPE and writes its record.

(require (for-syntax racket/base)
         (only-in json read-json write-json)
         (only-in racket/cmdline command-line)
         (only-in racket/list remove-duplicates)
         (only-in racket/math exact-round)
         (only-in racket/port port->string)
         (only-in racket/runtime-path define-runtime-path)
         (only-in racket/string string-join string-split)
         (only-in torch shape sum)
         (only-in torch/data/loader dataloader)
         (only-in "diag-counters.rkt"
                  counter-delta counter-snapshot ledger-entries load-average
                  median-record peak-rss-kb)
         (only-in "workloads/pipelines.rkt" part-timings pipeline))

(define-runtime-path twin "python/pipelines.py")
(define-runtime-path self "scope-pipelines.rkt")

;; WARMUP_EPOCHS and TIMED_EPOCHS reach both sides' children
(define (env-epochs name default)
  (or (let ([v (getenv name)]) (and v (string->number v))) default))
(define warmup-epochs (env-epochs "WARMUP_EPOCHS" 2))
(define timed-epochs (env-epochs "TIMED_EPOCHS" 3))
(define child-timeout-s 1200)

(define (epoch! loader)
  (for/sum ([vs (in-values-sequence loader)])
    (sum (car vs))
    (car (shape (car vs)))))

(define (run-cell name workers)
  (define-values (ds batch-size _per-epoch) (pipeline name))
  (define loader
    (if (zero? workers)
        (dataloader ds #:batch-size batch-size #:shuffle? #t)
        (dataloader ds #:batch-size batch-size #:shuffle? #t #:workers workers)))
  (for ([_ (in-range warmup-epochs)]) (epoch! loader))
  (define peak (box (ledger-entries)))
  (define sampler
    (thread (lambda ()
              (let loop ()
                (set-box! peak (max (unbox peak) (ledger-entries)))
                (sleep 0.02)
                (loop)))))
  (define s0 (counter-snapshot))
  (define parts0 (part-timings))
  (define n (for/sum ([_ (in-range timed-epochs)]) (epoch! loader)))
  (define delta (counter-delta s0 (counter-snapshot)))
  (define parts
    (for/list ([kv (in-list (part-timings))])
      (cons (car kv) (- (cdr kv) (cdr (assq (car kv) parts0))))))
  (define part-samples (cdr (assq 'part-samples parts)))
  (kill-thread sampler)
  (define elapsed (cdr (assq 'elapsed-s delta)))
  (append (list (cons 'samples n)
                (cons 'samples/s (/ n elapsed))
                (cons 'peak-ledger-entries (unbox peak))
                (cons 'peak-rss-mb (let ([kb (peak-rss-kb)]) (and kb (quotient kb 1024)))))
          (if (zero? part-samples)
              '()
              (for/list ([kv (in-list parts)] #:unless (eq? (car kv) 'part-samples))
                (cons (string->symbol (format "~a/sample" (car kv)))
                      (/ (cdr kv) (exact->inexact part-samples)))))
          delta))

(define (child-env arm)
  (define env (environment-variables-copy (current-environment-variables)))
  (when arm
    (environment-variables-set! env #"RKTORCH_DIAG_SCOPE"
                                (string->bytes/utf-8 (symbol->string arm))))
  (environment-variables-set! env #"OMP_NUM_THREADS" #"1")
  (environment-variables-set! env #"MKL_NUM_THREADS" #"1")
  env)

;; a record, or (hash 'error text) when the child failed or timed out
(define (run-child program args arm)
  (define-values (sub out in err)
    (parameterize ([current-environment-variables (child-env arm)])
      (apply subprocess #f #f #f program args)))
  (close-output-port in)
  (define (reader port)
    (define text (box ""))
    (values text (thread (lambda () (set-box! text (port->string port))))))
  (define-values (out-text out-reader) (reader out))
  (define-values (err-text err-reader) (reader err))
  (define done (sync/timeout child-timeout-s sub))
  (unless done (subprocess-kill sub #t))
  (thread-wait out-reader)
  (thread-wait err-reader)
  (close-input-port out)
  (close-input-port err)
  (define record
    (with-handlers ([exn:fail? (lambda (_) #f)])
      (read-json (open-input-string (unbox out-text)))))
  (cond
    [(and done (zero? (subprocess-status sub)) (hash? record)) record]
    [else
     (define s (unbox err-text))
     (define tail (substring s (max 0 (- (string-length s) 2000))))
     (hash 'error (if done tail (string-append "timed out\n" tail)))]))

(define (run-side side name workers)
  (case side
    [(pytorch)
     (run-child (find-executable-path "python3")
                (list "-P" (path->string twin) "--cell"
                      (symbol->string name) (number->string workers))
                #f)]
    [else
     (run-child (find-executable-path (find-system-path 'exec-file))
                (list (path->string self) "--cell"
                      (symbol->string name) (number->string workers))
                side)]))

(define (rotate xs k)
  (define n (modulo k (length xs)))
  (append (list-tail xs n) (for/list ([x (in-list xs)] [_ (in-range n)]) x)))

(define (drive out-path names workers-list arms reps)
  (define sides (cons 'pytorch arms))
  (call-with-output-file out-path #:exists 'append
    (lambda (out)
      (for* ([name (in-list names)] [workers (in-list workers-list)]
             [rep (in-range reps)] [side (in-list (rotate sides rep))])
        (define load (load-average))
        (define record (run-side side name workers))
        (define line
          (hash-set* record 'pipeline (symbol->string name) 'workers workers
                     'side (symbol->string side) 'rep rep 'load load))
        (write-json line out)
        (newline out)
        (flush-output out)
        (printf "~a w=~a ~a rep ~a load ~a: ~a\n" name workers side rep load
                (if (hash-ref record 'error #f)
                    (format "ERROR ~a" (hash-ref record 'error))
                    (show (hash-ref record 'samples/s #f))))
        (flush-output)))))

(define (show v)
  (cond
    [(not v) "-"]
    [(and (real? v) (>= (abs v) 100)) (number->string (exact-round v))]
    [(real? v) (real->decimal-string v 2)]
    [else (format "~a" v)]))

(define (read-runs path)
  (call-with-input-file path
    (lambda (in)
      (for/list ([line (in-lines in)] #:unless (string=? line ""))
        (read-json (open-input-string line))))))

(define (medians runs pipeline workers side)
  (define rs
    (for/list ([r (in-list runs)]
               #:when (and (equal? (hash-ref r 'pipeline) pipeline)
                           (equal? (hash-ref r 'workers) workers)
                           (equal? (hash-ref r 'side) side)
                           (not (hash-ref r 'error #f))))
      (for/list ([(k v) (in-hash r)] #:when (real? v)) (cons k v))))
  (and (pair? rs) (cons (length rs) (median-record rs))))

(define (field m k)
  (cond
    [m
     (define kv (assq k (cdr m)))
     (and kv (cdr kv))]
    [else #f]))

(define (spread runs pipeline workers side)
  (define vs
    (for/list ([r (in-list runs)]
               #:when (and (equal? (hash-ref r 'pipeline) pipeline)
                           (equal? (hash-ref r 'workers) workers)
                           (equal? (hash-ref r 'side) side)
                           (hash-ref r 'samples/s #f)))
      (hash-ref r 'samples/s)))
  (if (< (length vs) 2)
      #f
      (/ (- (apply max vs) (apply min vs)) (/ (apply + vs) (length vs)))))

(define (summarize path)
  (define runs (read-runs path))
  (define (distinct key) (remove-duplicates (map (lambda (r) (hash-ref r key)) runs)))
  (for ([pipeline (in-list (distinct 'pipeline))])
    (define workers-list (sort (distinct 'workers) <))
    (define arms (filter (lambda (s) (not (equal? s "pytorch"))) (distinct 'side)))
    (printf "\n## ~a (samples/s, median of runs; spread = (max-min)/mean)\n\n" pipeline)
    (printf "| workers | PyTorch | ~a | best/PyTorch |\n" (string-join arms " | "))
    (printf "|---|---|~a---|\n" (apply string-append (map (lambda (_) "---|") arms)))
    (for ([w (in-list workers-list)])
      (define (cell side)
        (define m (medians runs pipeline w side))
        (define sp (spread runs pipeline w side))
        (if m
            (format "~a (n=~a, ±~a%)" (show (field m 'samples/s)) (car m)
                    (if sp (exact-round (* 100 sp)) "-"))
            "-"))
      (define py (field (medians runs pipeline w "pytorch") 'samples/s))
      (define best
        (for/fold ([b #f]) ([a (in-list arms)])
          (define v (field (medians runs pipeline w a) 'samples/s))
          (if (and v (or (not b) (> v b))) v b)))
      (printf "| ~a | ~a | ~a | ~a |\n" w (cell "pytorch")
              (string-join (map cell arms) " | ")
              (if (and py best) (real->decimal-string (/ best py) 2) "-")))
    (displayln
     "\n| workers | arm | finalizer runs | on main | exit-frees | escaped-to-finalizer | peak RSS MB | GC ms | load |")
    (displayln "|---|---|---|---|---|---|---|---|---|")
    (for* ([w (in-list workers-list)] [a (in-list arms)])
      (define m (medians runs pipeline w a))
      (when m
        (printf "| ~a | ~a | ~a | ~a | ~a | ~a | ~a | ~a | ~a |\n" w a
                (show (field m 'finalizer-runs)) (show (field m 'finalizer-main-runs))
                (show (field m 'exit-frees)) (show (field m 'escaped-to-finalizer))
                (show (field m 'peak-rss-mb)) (show (field m 'gc-ms))
                (string-join
                 (for/list ([r (in-list runs)]
                            #:when (and (equal? (hash-ref r 'pipeline) pipeline)
                                        (equal? (hash-ref r 'workers) w)
                                        (equal? (hash-ref r 'side) a)))
                   (car (string-split (hash-ref r 'load) " ")))
                 "/"))))
    (for ([r (in-list runs)] #:when (and (equal? (hash-ref r 'pipeline) pipeline)
                                         (hash-ref r 'error #f)))
      (printf "\nerror ~a w=~a ~a rep ~a:\n~a\n" pipeline (hash-ref r 'workers)
              (hash-ref r 'side) (hash-ref r 'rep) (hash-ref r 'error)))))

(module+ main
  (define names '(jpeg cifar text))
  (define workers-list '(0 1 2 4 6 8))
  (define arms '(off build ref))
  (define reps 3)
  (define out-path #f)
  (define summary #f)
  (define cell #f)
  (define (symbols s) (map string->symbol (string-split s ",")))
  (command-line
   #:once-each
   [("--out") p "JSON-lines file the runs are appended to" (set! out-path p)]
   [("--pipelines") p "Comma-separated pipelines" (set! names (symbols p))]
   [("--workers") w "Comma-separated worker counts"
                  (set! workers-list (map string->number (string-split w ",")))]
   [("--arms") a "Comma-separated arms" (set! arms (symbols a))]
   [("--reps") n "Repetitions per cell" (set! reps (string->number n))]
   [("--summarize") p "Print the median tables of a runs file" (set! summary p)]
   [("--cell") name workers "Run one rktorch cell in this process"
               (set! cell (list (string->symbol name) (string->number workers)))])
  (cond
    [cell
     (write-json (for/hasheq ([kv (in-list (run-cell (car cell) (cadr cell)))])
                   (values (car kv) (cdr kv))))
     (newline)]
    [summary (summarize summary)]
    [out-path (drive out-path names workers-list arms reps) (summarize out-path)]
    [else (raise-user-error 'scope-pipelines "give --out, --summarize or --cell")]))
