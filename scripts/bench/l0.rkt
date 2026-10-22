#lang racket/base

(require (only-in racket/format ~a)
         (only-in racket/list take-right)
         (only-in racket/match match)
         (only-in racket/string string-join)
         (only-in threading ~>)
         (only-in torch/tests/private/wedge call-with-process-deadline)
         (only-in "calls.rkt" call-records)
         (only-in "harness.rkt"
                  host-meta load-average measure result->record result-name result-per-call)
         (only-in "host-gpu.rkt" host-gpu-records)
         (only-in "pytorch.rkt" require-pytorch twin-record twin-records)
         (only-in "shim.rkt" aten-threads)
         (only-in "threads.rkt" main-states scaling-cases thread-counts)
         (only-in "workloads/phases.rkt" mnist-phases resnet-phases))

(provide l0-suites
         run-l0-suite)

(define l0-suites '(pytorch-ops threads pytorch-threads calls host-gpu))

(define scaling-deadline 3600)

(define proxy-base 4)

(define (scaling-records results runs meta settings)
  (for/list ([r (in-list results)])
    (define-values (state n) (name->case (result-name r)))
    (define timed
      (~> (hash-ref runs (cons state n) '()) reverse (take-right (length (result-per-call r)))))
    (define (of key) (map (lambda (run) (hash-ref run key)) timed))
    (result->record r #:suite "scaling" #:group "threads" #:unit "ops/s" #:meta meta #:scale 1
                    #:extra (hasheq 'impl "rktorch" 'case (~a state) 'variant (~a n)
                                    'threads_run n 'main_steps_per_s (of 'main_steps_per_s)
                                    'stopped (of 'stopped) 'settings settings))))

(define (comma-list xs)
  (string-join (map ~a xs) ","))

(define (name->case name)
  (match (regexp-match #rx"^([^/]*)/([0-9]+)$" (symbol->string name))
    [(list _ state n) (values (string->symbol state) (string->number n))]))

(define (run-l0-suite suite
                      #:emit emit
                      #:log log
                      #:device device
                      #:scale scale
                      #:rounds rounds
                      #:warmup warmup
                      #:libtorch libtorch
                      #:seconds [seconds (max 0.5 (* 2.0 scale))]
                      #:threads [threads thread-counts]
                      #:phases [phases (hasheq)])
  (define (meta-on d) (host-meta #:device d #:libtorch libtorch #:threads (aten-threads)))
  (define scaling-warmup (max 1 warmup))
  (define scaling-settings (hasheq 'seconds seconds 'warmup scaling-warmup))
  (define (mnist) (mnist-phases device #:batch (hash-ref phases 'batch 128)
                                #:samples (hash-ref phases 'samples 1024)))
  (define (resnet #:base [base (hash-ref phases 'base 64)])
    (resnet-phases device #:batch (hash-ref phases 'batch 128)
                   #:samples (hash-ref phases 'samples 1024)
                   #:base base))
  (case suite
    [(pytorch-ops)
     (require-pytorch 'pytorch-ops)
     (define load (load-average))
     (for ([child (in-list (twin-records "ops.py"
                                         (list "--device" device "--scale" scale
                                               "--rounds" (or rounds 5) "--warmup" warmup
                                               "--threads" (aten-threads))
                                         #:log log))])
       (emit (twin-record child #:suite "micro" #:meta (meta-on device) #:load load)))]
    [(threads)
     (define runs (make-hash))
     (define (on-run state n run) (hash-update! runs (cons state n) (lambda (rs) (cons run rs)) '()))
     (define results
       (call-with-process-deadline
        scaling-deadline
        (lambda ()
          (measure (scaling-cases #:seconds seconds #:threads threads #:on-run on-run)
                   #:warmup scaling-warmup #:rounds (or rounds 5)))))
     (for-each emit (scaling-records results runs (meta-on 'cpu) scaling-settings))]
    [(pytorch-threads)
     (require-pytorch 'pytorch-threads)
     (define load (load-average))
     (for ([child (in-list (twin-records "threads.py"
                                         (list "--seconds" seconds "--rounds" (or rounds 5)
                                               "--warmup" scaling-warmup
                                               "--threads" (comma-list threads)
                                               "--states" (comma-list main-states)
                                               "--intra-op" (aten-threads))
                                         #:log log))])
       (emit (~> (twin-record child #:suite "scaling" #:meta (meta-on 'cpu) #:load load)
                 (hash-set 'settings scaling-settings))))]
    [(calls)
     (define meta (meta-on device))
     (for* ([p (in-list (list (mnist) (resnet)))]
            [r (in-list (call-records p #:rounds (or rounds 3)))])
       (emit (hash-set* r 'meta meta 'load (load-average))))]
    [(host-gpu)
     (define meta (meta-on device))
     (for ([r (in-list (host-gpu-records (resnet) #:proxy (resnet #:base proxy-base)
                                         #:steps (~> scale (* 20) round inexact->exact (max 2))
                                         #:rounds (or rounds 5) #:warmup (max 1 warmup)))])
       (emit (hash-set* r 'meta meta 'load (load-average))))]))

(module+ test
  (require (only-in racket/list remove-duplicates)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "ops.rkt" op-names)
           (only-in "pytorch.rkt" pytorch-available?))

  (define (records-of suite #:scale [scale 0.001] #:warmup [warmup 0])
    (define out '())
    (run-l0-suite suite #:emit (lambda (r) (set! out (cons r out))) #:log (open-output-nowhere)
                  #:device 'cpu #:scale scale #:rounds 1 #:warmup warmup #:libtorch "x"
                  #:threads '(1 2) #:phases (hasheq 'batch 2 'samples 8 'base 4))
    (reverse out))

  (define (shape-of r)
    (list (hash-ref r 'suite) (hash-ref r 'group) (hash-ref r 'impl)))

  (define (cases-of records)
    (map (lambda (r) (list (hash-ref r 'case) (hash-ref r 'variant))) records))

  (define scaling-floor (hasheq 'seconds 0.5 'warmup 1))

  (test-case "threads records ops per second per main state and thread count"
    (define records (records-of 'threads))
    (check-equal? (cases-of records) '(("idle" "1") ("idle" "2") ("train" "1") ("train" "2")))
    (for ([r (in-list records)])
      (check-equal? (shape-of r) '("scaling" "threads" "rktorch"))
      (check-equal? (hash-ref r 'unit) "ops/s")
      (check-equal? (length (hash-ref r 'main_steps_per_s)) 1 "the warmup run is not kept")
      (check-equal? (hash-ref r 'stopped) (list (hash-ref r 'threads_run)))
      (check-equal? (hash-ref r 'settings) scaling-floor "a smoke run still times whole seconds")
      (check-true (>= (hash-ref (hash-ref r 'stats) 'median) 0))))

  (test-case "calls and host-gpu record every phase of 05 and 09"
    (define calls (records-of 'calls))
    (check-equal? (remove-duplicates (map (lambda (r) (hash-ref r 'case)) calls))
                  '("05-mnist" "09-resnet"))
    (for ([r (in-list calls)])
      (check-equal? (shape-of r) '("calls" "phase" "rktorch"))
      (check-equal? (hash-ref (hash-ref r 'meta) 'device) "cpu"))
    (define timed (records-of 'host-gpu))
    (check-equal? (remove-duplicates (map (lambda (r) (hash-ref r 'group)) timed))
                  '("step" "phase"))
    (check-equal? (for/list ([r (in-list timed)] #:when (equal? (hash-ref r 'variant) "proxy"))
                    (hash-ref (hash-ref r 'settings) 'base))
                  (list proxy-base))
    (check-true (andmap (lambda (r) (equal? (hash-ref r 'case) "09-resnet")) timed)))

  (cond
    [(not (pytorch-available?))
     (displayln
      "[l0] PyTorch twins skipped: python3 `torch` not available (run inside `nix develop`)")]
    [else
     (test-case "pytorch-ops records every op of the ops suite, as PyTorch"
       (define records (records-of 'pytorch-ops))
       (check-equal? (map (lambda (r) (hash-ref r 'case)) records) (map symbol->string op-names))
       (for ([r (in-list records)])
         (define meta (hash-ref r 'meta))
         (check-equal? (shape-of r) '("micro" "ops" "pytorch"))
         (check-equal? (hash-ref r 'variant) "pytorch")
         (check-equal? (hash-ref meta 'threads) (aten-threads))
         (check-equal? (hash-ref meta 'threads_source) "torch.get_num_threads")
         (check-true (string? (hash-ref meta 'git_sha)) "the parent completes the meta")
         (check-true (positive? (hash-ref (hash-ref r 'stats) 'median)))))

     (test-case "pytorch-threads records the cases threads does"
       (define records (records-of 'pytorch-threads))
       (check-equal? (cases-of records) '(("idle" "1") ("idle" "2") ("train" "1") ("train" "2")))
       (for ([r (in-list records)])
         (check-equal? (shape-of r) '("scaling" "threads" "pytorch"))
         (check-equal? (hash-ref r 'settings) scaling-floor)
         (check-true (positive? (hash-ref (hash-ref r 'stats) 'median)))))

     (test-case "pytorch-threads counts the mixes done when main ends, from when the workers go"
       (match (twin-records "threads_probe.py" '() #:log (open-output-nowhere) #:time-limit 120)
         [(list slow stamped)
          (check-equal? (hash-ref slow 'ops_per_s) 0.0 "a mix finished after the run is not counted")
          (check-equal? (hash-ref slow 'stopped) 1)
          (check-equal? (hash-ref stamped 'workers) 4)
          (check-equal? (hash-ref stamped 'before_start) 0)]))]))
