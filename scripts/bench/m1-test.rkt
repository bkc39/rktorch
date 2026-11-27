#lang racket/base

(module+ test
  (require (only-in racket/file make-temporary-directory)
           (only-in racket/list remove-duplicates)
           (only-in racket/string string-contains?)
           rackunit
           (only-in "cifar-folder.rkt" ensure-cifar10-folder!)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "m1.rkt" m1-arms m1-plan m1-settings)
           (only-in "m1-cell.rkt" handoff-cell span-split)
           (only-in "m1-clocks.rkt" gc-event span)
           (only-in "summary-m1.rkt" gc-row loss-row m1-markdown registration-rows valid-arms))

  (define (plan #:seed [seed 1])
    (m1-plan #:rounds 2 #:pipelines '(cifar-png jpeg) #:workers '(0 4) #:trips '(#f 32)
             #:seed seed))

  (test-case "a round runs every cell once; a trip cell runs rktorch alone, a default one its twin"
    (define blocks (plan))
    (check-equal? (length blocks) 16)
    (check-equal? (plan) blocks)
    (check-not-equal? (plan #:seed 2) blocks)
    (check-equal? (map (lambda (args) (apply m1-arms args)) '((0 #f) (4 #f) (4 32) (0 64)))
                  '((rktorch pytorch) (rktorch pytorch-persistent) (rktorch) (rktorch))))

  (test-case "settings read trips with - as Racket's own, and paths as a summary request"
    (define s (m1-settings (vector "--trips" "-,32" "--workers" "0,7" "a.jsonl")))
    (check-equal? (hash-ref s 'trips) '(#f 32))
    (check-equal? (hash-ref s 'workers) '(0 7))
    (check-equal? (hash-ref s 'summarize) '("a.jsonl")))

  (define spans-in
    (map (lambda (fields) (apply span fields))
         '((epoch 1 0.0 100.0 0.0 1.0)
           (next 1 0.0 60.0 0.0 1.0)
           (step 1 60.0 70.0 1.0 2.0)
           (build 2 10.0 40.0 0.0 20.0)
           (build 2 50.0 90.0 20.0 50.0)
           (build 3 20.0 80.0 0.0 50.0))))

  (test-case "worker idle splits into the head, the gaps between builds, the tail and absent workers"
    (define split (span-split spans-in (list (gc-event 'minor 0 0 1 30.0 35.0 #f))
                              #:main 1 #:workers 3))
    (check-equal? (hash-ref split 'worker_idle)
                  (hasheq 'head_ms 30.0 'tail_ms 30.0 'gaps_ms 10.0 'absent_ms 100.0))
    (check-equal? (hash-ref (hash-ref split 'worker_build) 'wall_ms) 130.0)
    (check-equal? (hash-ref (hash-ref split 'worker_build) 'cpu_ms) 100.0)
    (check-equal? (hash-ref (hash-ref split 'worker_build) 'gc_ms) 10.0
                  "a collection stalls every build it overlaps")
    (check-equal? (hash-ref split 'worker_threads) 2))

  (define (arm impl workers us #:block block #:outside [outside 0.5] #:extra [extra (hasheq)])
    (for/fold ([r (hasheq 'kind "arm" 'run_id "r" 'round 0 'block block 'impl impl
                          'pipeline "jpeg" 'workers workers 'us_per_sample us
                          'samples_per_s (/ 1e6 us) 'load_outside outside 'max_load 2.0
                          'tree "master" 'racket "9.3")])
              ([(k v) (in-hash extra)])
      (hash-set r k v)))

  (define (split-of build-kind wall cpu gc #:idle [idle (hasheq)])
    (define (t w c g) (hasheq 'wall_ms w 'cpu_ms c 'gc_ms g 'count 1))
    (hasheq 'samples 10 'main_cpu_ms 1.0 'trip_mb #f
            'gc (hasheq 'all (hasheq 'count 2 'real_ms 1.0 'cpu_ms 1.0 'mean_live 50.0
                                     'mean_real_ms 0.5)
                        'minor (hasheq 'count 2 'mean_real_ms 0.5)
                        'major (hasheq 'count 0 'mean_real_ms #f)
                        'points '(("minor" 0.4 10) ("minor" 0.5 20) ("minor" 0.6 30)))
            'split (hasheq build-kind (t wall cpu gc)
                           'main_collate (t 0.0 0.0 0.0) 'worker_collate (t 0.0 0.0 0.0)
                           'step (t 1.0 1.0 0.0) 'next (t 1.0 0.5 0.0)
                           'start (t 0.0 0.0 0.0) 'finish (t 0.0 0.0 0.0)
                           'worker_idle (hash-union* (hasheq 'head_ms 0.0 'tail_ms 0.0
                                                             'gaps_ms 0.0 'absent_ms 0.0)
                                                     idle))))

  (define (hash-union* a b)
    (for/fold ([a a]) ([(k v) (in-hash b)]) (hash-set a k v)))

  (define records
    (list (arm "rktorch" 0 1000.0 #:block 0 #:extra (split-of 'main_build 9.0 8.0 0.5))
          (arm "pytorch" 0 800.0 #:block 0)
          (arm "rktorch" 2 700.0 #:block 1
               #:extra (split-of 'worker_build 12.0 10.0 1.0 #:idle (hasheq 'tail_ms 2.0)))
          (arm "pytorch" 2 420.0 #:block 1)
          (arm "rktorch" 2 9.0 #:block 2 #:outside 2.5)
          (arm "pytorch" 2 9.0 #:block 2)))

  (test-case "a block with an arm over the outside load is dropped whole, at its own or a given limit"
    (define-values (kept dropped) (valid-arms records))
    (check-equal? dropped 1)
    (check-equal? (length kept) 4)
    (define-values (all none) (valid-arms records #:limit 3.0))
    (check-equal? (list (length all) none) '(6 0)))

  (test-case "the loss terms sum to the measured loss: build, idle and the serial w0 work"
    (define-values (kept _dropped) (valid-arms records))
    (define row (loss-row kept "jpeg" 2))
    (check-equal? (hash-ref row 'loss) 200.0)
    (check-equal? (hash-ref row 'py_loss) 20.0)
    (check-= (hash-ref row 'explained) (hash-ref row 'loss) 1e-9)
    (check-= (hash-ref row 'cpu) 100.0 1e-9)
    (check-= (hash-ref row 'tail) 100.0 1e-9))

  (test-case "GC rows count collections per thousand samples and fit ms against live entries"
    (define-values (kept _dropped) (valid-arms records))
    (define row (gc-row kept "jpeg" 0 #f))
    (check-equal? (hash-ref row 'minor_per_k) 200.0)
    (check-= (hash-ref row 'us_per_k_live) 10.0 1e-9)
    (check-true (string-contains? (m1-markdown records) "or a failed arm: 1")))

  (test-case "registration pieces are differences of the mix's layers"
    (define (op layer n us)
      (hasheq 'kind "op" 'op "mix" 'layer layer 'threads n 'us_per_op us 'ops_per_call 4
              'heap_bytes_per_op 4096.0 'gc_real_ms 0.0 'ops 100))
    (define-values (threads _layers pieces _chain)
      (registration-rows (list (op "raw" 1 4.0) (op "registered" 1 12.0) (op "guardian" 1 8.0))))
    (check-equal? threads '(1))
    (check-equal? (assoc "registration" pieces) '("registration" (2.0 2.0 0.0)))
    (check-equal? (assoc "guardian" pieces) '("guardian" (1.0 1.0 0.0))))

  (test-case "a cell on the PNG fixture splits main's and the workers' time, at tiny sizes"
    (with-temporary-path (base (make-temporary-directory "m1-test-~a"))
      (ensure-cifar10-folder! "fixture" #:base base)
      (parameterize ([current-environment-variables
                      (environment-variables-copy (current-environment-variables))])
        (putenv "RKTORCH_CIFAR10_PNG_DIR" (path->string base))
        (for ([w (in-list '(0 2))])
          (define r (handoff-cell 'cifar-png w #:warmup 0 #:epochs 1 #:samples 32
                                  #:dataset "fixture" #:count? (zero? w)))
          (check-equal? (hash-ref r 'samples) 32)
          (define split (hash-ref r 'split))
          (define build (hash-ref split (if (zero? w) 'main_build 'worker_build)))
          (check-true (positive? (hash-ref build 'wall_ms)))
          (check-equal? (hash-ref split 'worker_threads) (if (zero? w) 0 1)
                        "a 32-sample epoch is one 128-sample batch")
          (when (zero? w)
            (check-true (positive? (hash-ref (hash-ref r 'counts) 'tensors_per_sample))))))))

  (test-case "every pipeline name the cell takes is a loader the driver plans"
    (check-equal? (remove-duplicates (map (lambda (b) (hash-ref b 'pipeline)) (plan)))
                  '(cifar-png jpeg))))
