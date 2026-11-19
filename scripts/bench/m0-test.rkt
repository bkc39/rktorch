#lang racket/base

(module+ test
  (require (only-in racket/file make-temporary-directory)
           (only-in racket/list count remove-duplicates)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "cifar-folder.rkt" ensure-cifar10-folder!)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "m0.rkt"
                    block-arms cell-args m0-paths m0-plan m0-settings racket-label run-arm
                    run-blocks)
           (only-in "m0-cells.rkt" gc-count loader-cell measured train-cell)
           (only-in "pytorch.rkt" pytorch-available?)
           (only-in "summary-m0.rkt" m0-markdown))

  (define quiet (open-output-nowhere))

  (define (plan #:rounds [rounds 3] #:seed [seed 1])
    (m0-plan #:rounds rounds #:pipelines '(cifar-png jpeg) #:workers '(0 1 4)
             #:train-workers '(1 8) #:seed seed))

  (define (cells-of blocks)
    (map (lambda (b) (cons (hash-ref b 'pipeline) (hash-ref b 'workers))) blocks))

  (define (in-round blocks r)
    (filter (lambda (b) (= (hash-ref b 'round) r)) blocks))

  (test-case "a round runs every cell once, each block every arm its cell has, in its own order"
    (define blocks (plan))
    (check-equal? (length blocks) 24)
    (for ([r (in-range 3)])
      (define round (in-round blocks r))
      (check-equal? (sort (cells-of round) string<? #:key (lambda (c) (format "~a" c)))
                    (sort '((cifar-png . 0) (cifar-png . 1) (cifar-png . 4) (jpeg . 0) (jpeg . 1)
                            (jpeg . 4) (train . 1) (train . 8))
                          string<? #:key (lambda (c) (format "~a" c))))
      (check-equal? (map (lambda (b) (hash-ref b 'block)) round) '(0 1 2 3 4 5 6 7)))
    (for ([b (in-list blocks)])
      (check-equal? (sort (hash-ref b 'arms) symbol<?)
                    (sort (block-arms (hash-ref b 'pipeline) (hash-ref b 'workers)) symbol<?)))
    (check-equal? (block-arms 'jpeg 0) '(rktorch pytorch) "no persistent workers at w0")
    (check-equal? (block-arms 'train 4) '(rktorch pytorch))
    (check-equal? (block-arms 'cifar-png 4) '(rktorch pytorch pytorch-persistent)))

  (test-case "the order is shuffled per round and per block, and the seed fixes it"
    (define blocks (plan))
    (check-equal? (plan) blocks)
    (check-not-equal? (plan #:seed 2) blocks)
    (check-true (> (length (remove-duplicates (for/list ([r (in-range 3)])
                                                (cells-of (in-round blocks r)))))
                   1))
    (check-true (positive? (count (lambda (b) (not (eq? (car (hash-ref b 'arms)) 'rktorch)))
                                  blocks))
                "rktorch does not always run first"))

  (test-case "a run names the Racket it must be, and refuses another"
    (check-equal? (racket-label #f "9.3") "9.3")
    (check-equal? (racket-label "9.1" "9.1.0.4") "9.1")
    (check-exn #rx"asked for Racket 9.1, but this is Racket 9.3"
               (lambda () (racket-label "9.1" "9.3")))
    (define-values (jsonl md base) (m0-paths "/tmp/b" "9.3" "2026-11-19"))
    (check-equal? (path->string jsonl) "/tmp/b/m0-9.3-2026-11-19.jsonl")
    (check-equal? (path->string md) "/tmp/b/m0-9.3-2026-11-19.md")
    (check-equal? base "m0-9.3-2026-11-19"))

  (test-case "settings: defaults, --smoke, and flags after it"
    (define defaults (m0-settings #()))
    (check-equal? (hash-ref defaults 'workers) '(0 1 2 4 7))
    (check-equal? (hash-ref defaults 'train-workers) '(1 2 4 8))
    (check-equal? (hash-ref defaults 'k3-reps) 5 "K3 runs as many reps as rounds")
    (check-false (hash-ref defaults 'summarize))
    (define smoke (m0-settings #("--smoke" "--max-load" "99" "--samples" "32")))
    (check-equal? (map (lambda (k) (hash-ref smoke k)) '(rounds samples dataset max-load k3-reps))
                  '(1 32 "fixture" 99 1))
    (check-equal? (hash-ref (m0-settings #("--train-workers" "-")) 'train-workers) '())
    (check-equal? (hash-ref (m0-settings #("a.jsonl" "b.jsonl")) 'summarize) '("a.jsonl" "b.jsonl"))
    (check-exn #rx"expected a count, got \"x\"" (lambda () (m0-settings #("--rounds" "x"))))
    (check-exn #rx"expected a count" (lambda () (m0-settings #("--workers" "0,-1")))))

  (define settings
    (hash-set* (m0-settings #("--samples" "16" "--epochs" "1" "--warmup" "0" "--seconds" "0.2"
                              "--dataset" "fixture"))
               'max-load 4.0))

  (test-case "an arm's child gets its cell, and only the persistent twin keeps its workers"
    (define block (hasheq 'round 0 'block 0 'pipeline 'jpeg 'workers 4 'arms '()))
    (check-equal? (cell-args block 'rktorch settings)
                  '("--warmup" 0 "--epochs" 1 "--samples" 16 "--dataset" "fixture" "loader" jpeg 4))
    (check-equal? (cell-args block 'pytorch-persistent settings)
                  '("--persistent" "--warmup" 0 "--epochs" 1 "--samples" 16 "--dataset" "fixture"
                                   "loader" jpeg 4))
    (check-equal? (cell-args (hash-set block 'pipeline 'train) 'pytorch settings)
                  '("--seconds" 0.2 "--warmup" 0 "train" 4))
    (check-equal? (list-ref (cell-args block 'rktorch (hash-set settings 'samples #f)) 5) 512
                  "each pipeline has its own sample count"))

  (test-case "an arm's record carries its block, the outside load, and an error in place of a child"
    (define block (hasheq 'round 2 'block 3 'pipeline 'cifar 'workers 1 'arms '()))
    (define ticks (box '((100 1000 40) (400 2000 240))))
    (define (cpu-times)
      (begin0 (car (unbox ticks)) (set-box! ticks (cdr (unbox ticks)))))
    (define r
      (run-arm block 'pytorch-persistent 1 settings
               #:run (lambda (_b _a _s #:log _log) (hasheq 'samples_per_s 10.0 'racket "x"))
               #:cpu-times cpu-times #:load (lambda () '(0.5 0.5 0.5)) #:log quiet))
    (check-equal? (for/list ([k (in-list '(kind round block order pipeline workers arm impl
                                                racket max_load samples_per_s))])
                    (hash-ref r k))
                  (list "arm" 2 3 1 "cifar" 1 "pytorch-persistent" "pytorch" (hash-ref settings 'racket)
                        4.0 10.0))
    (check-true (real? (hash-ref r 'load_outside)))
    (define failed
      (run-arm block 'rktorch 0 settings
               #:run (lambda (_b _a _s #:log _log) (error 'cell "no tree"))
               #:cpu-times (lambda () #f) #:log quiet))
    (check-regexp-match #rx"no tree" (hash-ref failed 'error))
    (check-equal? (hash-ref failed 'impl) "rktorch")
    (check-false (hash-ref failed 'load_outside)))

  (test-case "blocks wait for the load first, then run their arms back to back in their order"
    (define events '())
    (define (note! e) (set! events (cons e events)))
    (define blocks (plan #:rounds 1))
    (define emitted '())
    (run-blocks blocks settings
                #:emit (lambda (r) (set! emitted (cons r emitted)))
                #:log quiet
                #:wait (lambda (limit _log) (note! (list 'wait limit)))
                #:run (lambda (b a _s #:log _log)
                        (note! (list (hash-ref b 'block) a))
                        (hasheq 'samples_per_s 1.0)))
    (define expected
      (apply append (for/list ([b (in-list blocks)])
                      (cons '(wait 4.0) (for/list ([a (in-list (hash-ref b 'arms))])
                                          (list (hash-ref b 'block) a))))))
    (check-equal? (reverse events) expected)
    (check-equal? (length emitted) (apply + (map (lambda (b) (length (hash-ref b 'arms))) blocks))))

  (test-case "measured counts the collections, the GC time and the OS threads a thunk ran"
    (call-with-process-deadline
     60
     (lambda ()
       (define-values (result stats)
         (measured (lambda ()
                     (define t (thread (lambda () (sleep 0.4)) #:pool 'own))
                     (collect-garbage)
                     (thread-wait t)
                     'done)))
       (check-equal? result 'done)
       (check-true (>= (hash-ref stats 'gc_count) 1))
       (check-true (> (gc-count) 0))
       (check-true (real? (hash-ref stats 'elapsed_s)))
       (when (directory-exists? "/proc/self/task")
         (check-true (> (hash-ref stats 'os_threads_peak) (hash-ref stats 'os_threads_baseline))
                     (format "~a" stats))))))

  (test-case "a loader cell and a training cell record their rates and the Racket that ran them"
    (call-with-process-deadline
     300
     (lambda ()
       (for ([w (in-list '(0 1))])
         (define r (loader-cell 'cifar w #:warmup 0 #:epochs 1 #:samples 16))
         (check-equal? (hash-ref r 'samples) 16)
         (check-true (positive? (hash-ref r 'samples_per_s)))
         (check-equal? (hash-ref r 'racket) (version)))
       (define t (train-cell 1 #:seconds 0.1 #:warmup 0))
       (check-true (positive? (hash-ref t 'main_steps_per_s)))
       (check-true (positive? (hash-ref t 'ops_per_s))))))

  (cond
    [(not (pytorch-available?))
     (displayln "[m0] blocks end to end skipped: python3 `torch` not available")]
    [else
     (test-case "every cell and arm of a small plan records, and the summary prints"
       (call-with-process-deadline
        600
        (lambda ()
          (with-temporary-path (base (make-temporary-directory "m0-png-~a"))
            (parameterize ([current-environment-variables
                            (environment-variables-copy (current-environment-variables))])
              (putenv "RKTORCH_CIFAR10_PNG_DIR" (path->string base))
              (ensure-cifar10-folder! "fixture" #:base base)
              (define blocks
                (m0-plan #:rounds 1 #:pipelines '(cifar-png cifar) #:workers '(0 1)
                         #:train-workers '(1) #:seed 3))
              (define records '())
              (define log (open-output-string))
              (run-blocks blocks settings #:emit (lambda (r) (set! records (cons r records)))
                          #:log log #:wait void)
              (check-equal? (length records) 12)
              (for ([r (in-list records)])
                (check-false (hash-ref r 'error #f) (format "~a\n~a" r (get-output-string log)))
                (check-true (positive? (or (hash-ref r 'samples_per_s #f)
                                           (hash-ref r 'main_steps_per_s)))))
              (define md (m0-markdown records))
              (for ([heading (in-list '("## Racket" "### Done when" "### Table H"
                                        "### #439"))])
                (check-regexp-match (regexp-quote heading) md)))))))]))
