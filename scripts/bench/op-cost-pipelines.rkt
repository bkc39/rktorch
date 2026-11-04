#lang racket/base

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in torch shape sum)
         (only-in torch/data/loader dataloader)
         (only-in torch/foreign/raw/collector finalizer-runs)
         (only-in "harness.rkt" clock-ms load-average rotate)
         (only-in "pytorch.rkt" child-records twin-records)
         (only-in "workloads/pipelines.rkt" pipeline))

(provide run-cell
         single-threaded-environment
         pipeline-settings
         pipeline-records)

(define-runtime-path self "op-cost.rkt")

(define (epoch! loader)
  (for/sum ([vs (in-values-sequence loader)])
    (sum (car vs))
    (car (shape (car vs)))))

(define (peak-rss-kb)
  (with-handlers ([exn:fail? (lambda (_e) #f)])
    (call-with-input-file "/proc/self/status"
      (lambda (in)
        (for/or ([line (in-lines in)])
          (define m (regexp-match #rx"^VmHWM:[ \t]*([0-9]+)" line))
          (and m (string->number (cadr m))))))))

(define (run-cell pipeline-name workers #:warmup warmup #:epochs epochs #:samples samples)
  (define-values (ds batch-size _n)
    (pipeline pipeline-name #:samples samples))
  (define loader
    (if (zero? workers)
        (dataloader ds #:batch-size batch-size #:shuffle? #t)
        (dataloader ds #:batch-size batch-size #:shuffle? #t #:workers workers)))
  (for ([_ (in-range warmup)]) (epoch! loader))
  (define gc0 (current-gc-milliseconds))
  (define runs0 (finalizer-runs))
  (define t0 (clock-ms))
  (define n (for/sum ([_ (in-range epochs)]) (epoch! loader)))
  (define elapsed-s (/ (- (clock-ms) t0) 1000.0))
  (hasheq 'samples n
          'samples_per_s (/ n elapsed-s)
          'elapsed_s elapsed-s
          'gc_ms (- (current-gc-milliseconds) gc0)
          'finalizer_runs (- (finalizer-runs) runs0)
          'peak_rss_mb (let ([kb (peak-rss-kb)]) (and kb (quotient kb 1024)))))

(define (pipeline-settings #:warmup warmup #:epochs epochs #:samples samples)
  (list "--warmup" warmup "--epochs" epochs "--samples" (or samples 0)))

(define (single-threaded-environment)
  (define env (environment-variables-copy (current-environment-variables)))
  (for ([k (in-list '(#"OMP_NUM_THREADS" #"MKL_NUM_THREADS"))])
    (environment-variables-set! env k #"1"))
  env)

(define (side-record side pipeline workers settings log time-limit)
  (define args (append (list "--cell" pipeline workers) settings))
  (define records
    (parameterize ([current-environment-variables (single-threaded-environment)])
      (case side
        [(pytorch) (twin-records "pipelines.py" args #:log log #:time-limit time-limit)]
        [else (child-records (find-exe) (cons self args)
                             #:log log #:time-limit time-limit #:who 'op-cost-cell)])))
  (car records))

(define (pipeline-records pipelines workers-list reps
                          #:settings settings #:emit emit #:log log
                          #:time-limit [time-limit 1800])
  (for* ([rep (in-range reps)] [pipeline (in-list pipelines)] [workers (in-list workers-list)]
         [side (in-list (rotate '(rktorch pytorch) rep))])
    (define load (load-average))
    (define record
      (with-handlers ([exn:fail? (lambda (e) (hasheq 'error (exn-message e)))])
        (side-record side pipeline workers settings log time-limit)))
    (fprintf log "pipeline ~a w=~a ~a rep ~a: ~a\n" pipeline workers side rep
             (hash-ref record 'samples_per_s (lambda () (hash-ref record 'error #f))))
    (emit (hash-set* record 'kind "pipeline" 'impl (symbol->string side)
                     'pipeline (symbol->string pipeline) 'workers workers 'rep rep
                     'load load 'load_end (load-average)))))
