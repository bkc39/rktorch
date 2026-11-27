#lang racket/base

(module+ test
  (require (only-in racket/file make-temporary-directory)
           (only-in racket/hash hash-union)
           (only-in racket/list remove-duplicates)
           (only-in racket/path path-only)
           (only-in racket/port open-output-nowhere)
           (only-in racket/string string-contains?)
           rackunit
           (only-in "cifar-folder.rkt" ensure-cifar10-folder!)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "m1.rkt" m1-plan m1-settings run-m1)
           (only-in "m1-cell.rkt" handoff-cell span-split)
           (only-in "m1-clocks.rkt" gc-event os-thread-id span spans timed with-gc-log)
           (only-in "summary-m1.rkt"
                    gc-row handoff-terms loss-row m1-arms m1-markdown registration-rows valid-arms))

  (define (plan #:seed [seed 1])
    (m1-plan #:rounds 2 #:pipelines '(cifar-png jpeg) #:workers '(0 4) #:trips '(#f 32)
             #:seed seed))

  (test-case "a round runs every cell once; a trip cell runs rktorch alone, a default one M0's arms"
    (define blocks (plan))
    (check-equal? (length blocks) 16)
    (check-equal? (plan) blocks)
    (check-not-equal? (plan #:seed 2) blocks)
    (check-equal? (map (lambda (args) (apply m1-arms args))
                       '((jpeg 0 #f) (jpeg 4 #f) (jpeg 4 32) (cifar-png 0 64)))
                  '((rktorch pytorch) (rktorch pytorch pytorch-persistent) (rktorch) (rktorch))))

  (test-case "settings read trips with - as Racket's own, paths as a summary request, and refuse junk"
    (define s (m1-settings (vector "--trips" "-,32" "--workers" "0,7" "a.jsonl")))
    (check-equal? (hash-ref s 'trips) '(#f 32))
    (check-equal? (hash-ref s 'workers) '(0 7))
    (check-equal? (hash-ref s 'summarize) '("a.jsonl"))
    (check-exn #rx"expected a count, got \"x\"" (lambda () (m1-settings (vector "--rounds" "x"))))
    (check-exn #rx"expected a positive number" (lambda () (m1-settings (vector "--trips" "0")))))

  (test-case "a run whose records would land inside the repository is refused before it starts"
    (define here (path-only (variable-reference->module-source (#%variable-reference))))
    (check-exn #rx"inside the repository"
               (lambda ()
                 (run-m1 (hash-set (m1-settings (vector)) 'out-dir (build-path here "m1-out"))
                         #:log (open-output-nowhere)))))

  (define spans-in
    (map (lambda (fields) (apply span fields))
         '((epoch 1 0.0 100.0 0.0 1.0)
           (next 1 0.0 60.0 0.0 1.0)
           (step 1 60.0 70.0 1.0 2.0)
           (build 2 10.0 40.0 0.0 20.0)
           (build 2 50.0 90.0 20.0 50.0)
           (build 3 20.0 80.0 0.0 50.0)
           (build 3 90.0 110.0 50.0 65.0))))

  (test-case "worker idle and the build clipped to the epoch cover every worker's epoch exactly"
    (define split (span-split spans-in (list (gc-event 'minor 0 0 1 30.0 35.0 #f))
                              #:main 1 #:workers 3))
    (define idle (hash-ref split 'worker_idle))
    (define build (hash-ref split 'worker_build))
    (check-equal? idle #hasheq((head_ms . 30.0) (tail_ms . 10.0) (gaps_ms . 20.0) (absent_ms . 100.0)))
    (check-equal? (map (lambda (k) (hash-ref build k)) '(wall_ms cpu_ms off_ms outside_ms gc_ms))
                  '(150.0 115.0 35.0 10.0 10.0))
    (check-equal? (+ (- (hash-ref build 'wall_ms) (hash-ref build 'outside_ms))
                     (for/sum ([v (in-hash-values idle)]) v))
                  (* 3 100.0)
                  "three workers' epochs are their clipped builds and their idle time")
    (check-equal? (hash-ref split 'worker_threads) 2))

  (test-case "a collection on the thread that runs it is inside that thread's CPU, so off-CPU stays >= 0"
    (spans #:clear? #t)
    (define-values (_v events) (with-gc-log () (timed 'build (lambda () (collect-garbage 'major)))))
    (define split (span-split (spans #:clear? #t) events #:main (os-thread-id)))
    (check-true (positive? (hash-ref (hash-ref split 'main_build) 'gc_ms)))
    (define terms
      (handoff-terms `#hasheq((workers . 0) (samples . 1) (us_per_sample . 1.0) (split . ,split))))
    (for ([k (in-list '(build_wall build_cpu build_off build_gc))])
      (check-true (>= (hash-ref terms k) -1000.0) (format "~a: ~a µs" k (hash-ref terms k))))
    (check-= (+ (hash-ref terms 'build_cpu) (hash-ref terms 'build_off))
             (hash-ref terms 'build_wall) 1e-6))

  (define (timing wall cpu gc #:outside [outside 0.0])
    `#hasheq((wall_ms . ,wall) (cpu_ms . ,cpu) (off_ms . ,(- wall cpu)) (gc_ms . ,gc) (count . 1)
             (outside_ms . ,outside)))

  (define (rktorch-fields build wall cpu gc #:outside [outside 0.0] #:idle [idle #hasheq()])
    (define none (timing 0.0 0.0 0.0))
    `#hasheq((samples . 10) (main_cpu_ms . 1.0) (trip_mb . #f)
             (gc . #hasheq((all . #hasheq((count . 2) (real_ms . 1.0) (cpu_ms . 1.0)
                                          (mean_live . 50.0) (mean_real_ms . 0.5)))
                           (minor . #hasheq((count . 2) (mean_real_ms . 0.5)))
                           (major . #hasheq((count . 0) (mean_real_ms . #f)))
                           (points . (("minor" 0.4 10) ("minor" 0.5 20) ("minor" 0.6 30)))))
             (split . ,(hash-set
                        `#hasheq((main_collate . ,none) (worker_collate . ,none)
                                 (step . ,(timing 1.0 1.0 0.0)) (next . ,(timing 1.0 0.5 0.0))
                                 (start . ,none) (finish . ,none)
                                 (worker_idle
                                  . ,(hash-union #hasheq((head_ms . 0.0) (tail_ms . 0.0)
                                                         (gaps_ms . 0.0) (absent_ms . 0.0))
                                                 idle #:combine (lambda (_a b) b))))
                        build (timing wall cpu gc #:outside outside)))))

  (define (arm name workers us #:block block #:outside [outside 0.5] #:fields [fields #hasheq()])
    (hash-union
     `#hasheq((kind . "arm") (run_id . "r") (round . 0) (block . ,block) (arm . ,name)
              (pipeline . "jpeg") (workers . ,workers) (us_per_sample . ,us)
              (samples_per_s . ,(/ 1e6 us)) (load_outside . ,outside) (max_load . 2.0)
              (tree . "master") (racket . "9.3"))
     fields))

  (define records
    (list (arm "rktorch" 0 1000.0 #:block 0 #:fields (rktorch-fields 'main_build 9.0 8.0 0.5))
          (arm "pytorch" 0 800.0 #:block 0)
          (arm "rktorch" 2 700.0 #:block 1
               #:fields (rktorch-fields 'worker_build 12.0 10.0 1.0 #:idle #hasheq((tail_ms . 2.0))))
          (arm "pytorch" 2 420.0 #:block 1)
          (arm "pytorch-persistent" 2 300.0 #:block 1)
          (arm "rktorch" 2 800.0 #:block 3
               #:fields (rktorch-fields 'worker_build 14.0 9.0 2.0 #:outside 1.0
                                        #:idle #hasheq((head_ms . 1.0) (gaps_ms . 1.0))))
          (arm "pytorch" 2 460.0 #:block 3)
          (arm "pytorch-persistent" 2 320.0 #:block 3)
          (arm "rktorch" 2 9.0 #:block 2 #:outside 2.5)
          (arm "pytorch" 2 9.0 #:block 2)
          (arm "pytorch-persistent" 2 9.0 #:block 2)
          (arm "rktorch" 2 9.0 #:block 4)
          (arm "pytorch" 2 9.0 #:block 4)))

  (test-case "a block is dropped whole for an arm over the outside load or a missing arm"
    (define-values (kept dropped) (valid-arms records))
    (check-equal? dropped 2)
    (check-equal? (length kept) 8)
    (define-values (all none) (valid-arms records #:limit 3.0))
    (check-equal? (list (length all) none) '(11 1)))

  (test-case "each block's loss is its terms plus a residual, summarised by the median of each"
    (define-values (kept _dropped) (valid-arms records))
    (define row (loss-row kept "jpeg" 2))
    (check-equal? (hash-ref row 'blocks) 2)
    (check-= (hash-ref row 'loss) 250.0 1e-9)
    (check-= (hash-ref row 'cpu) 75.0 1e-9)
    (check-= (hash-ref row 'off) 125.0 1e-9)
    (check-= (hash-ref row 'outside) -25.0 1e-9)
    (check-= (hash-ref row 'serial) -50.0 1e-9)
    (check-= (hash-ref row 'residual) 25.0 1e-9 "block 1 leaves 0 µs, block 3 leaves 50 µs")
    (check-= (hash-ref row 'gc) 50.0 1e-9)
    (check-= (hash-ref row 'py_loss) 40.0 1e-9)
    (check-= (hash-ref row 'py_persistent_loss) -90.0 1e-9))

  (test-case "w0's off-CPU reference is its median wall less its median CPU, so a covered block leaves 0"
    (define w0
      (for/list ([block (in-list '(10 11 12))] [wall (in-list '(9.0 10.0 20.0))]
                 [cpu (in-list '(8.0 6.0 19.0))])
        (list (arm "rktorch" 0 (* 100.0 block) #:block block
                   #:fields (rktorch-fields 'main_build wall cpu 0.0))
              (arm "pytorch" 0 800.0 #:block block))))
    (define w2
      (list (arm "rktorch" 2 700.0 #:block 13
                 #:fields (rktorch-fields 'worker_build 12.0 10.0 0.0 #:idle #hasheq((tail_ms . 2.0))))
            (arm "pytorch" 2 420.0 #:block 13)
            (arm "pytorch-persistent" 2 300.0 #:block 13)))
    (define-values (kept _dropped) (valid-arms (apply append w2 w0)))
    (check-= (hash-ref (loss-row kept "jpeg" 2) 'residual) 0.0 1e-9))

  (test-case "GC rows count collections per thousand samples and fit ms against live entries"
    (define-values (kept _dropped) (valid-arms records))
    (define row (gc-row kept "jpeg" 0 #f))
    (check-equal? (hash-ref row 'minor_per_k) 200.0)
    (check-= (hash-ref row 'us_per_k_live) 10.0 1e-9)
    (define text (m1-markdown records))
    (check-true (string-contains? text "a failed or a missing arm: 2"))
    (check-true (string-contains? text "| residual |"))
    (check-true (string-contains? text "| jpeg | 2 | 2 | 750 | 500 | 250 |")))

  (test-case "registration pieces are differences against the layer that pays the same free"
    (define (op layer us)
      `#hasheq((kind . "op") (op . "mix") (layer . ,layer) (threads . 1) (us_per_op . ,us)
               (ops_per_call . 4) (heap_bytes_per_op . 4096.0) (gc_real_ms . 0.0) (ops . 100)))
    (define-values (threads _layers pieces _chain)
      (registration-rows (list (op "kept" 3.0) (op "raw" 4.0) (op "registered" 12.0)
                               (op "guardian" 8.0) (op "lifetimes" 6.0))))
    (check-equal? threads '(1))
    (check-equal? (map (lambda (name) (assoc name pieces)) '("free" "guardian" "lifetimes" "registration"))
                  '(("free" (0.25 0.25 0.0)) ("guardian" (1.25 1.25 0.0)) ("lifetimes" (0.5 0.5 0.0))
                    ("registration" (2.25 2.25 0.0)))))

  (test-case "a cell on the PNG fixture splits main's and the workers' time, at tiny sizes"
    (with-temporary-path (base (make-temporary-directory "m1-test-~a"))
      (ensure-cifar10-folder! "fixture" #:base base)
      (parameterize ([current-environment-variables
                      (environment-variables-copy (current-environment-variables))])
        (putenv "RKTORCH_CIFAR10_PNG_DIR" (path->string base))
        (for ([w (in-list '(0 2))])
          (define r (handoff-cell 'cifar-png w #:warmup 0 #:epochs 1 #:samples 32
                                  #:dataset "fixture"))
          (check-equal? (hash-ref r 'samples) 32)
          (define split (hash-ref r 'split))
          (define build (hash-ref split (if (zero? w) 'main_build 'worker_build)))
          (check-true (positive? (hash-ref build 'wall_ms)))
          (check-true (>= (hash-ref build 'off_ms) -1.0))
          (check-equal? (hash-ref split 'worker_threads) (if (zero? w) 0 1)
                        "a 32-sample epoch is one 128-sample batch")
          (check-equal? (and (hash-ref r 'counts) #t) (zero? w) "the counting pass runs at w0 only")
          (when (positive? w)
            (check-= (+ (- (hash-ref build 'wall_ms) (hash-ref build 'outside_ms))
                        (for/sum ([v (in-hash-values (hash-ref split 'worker_idle))]) v))
                     (* w (hash-ref (hash-ref split 'epoch) 'wall_ms))
                     1e-6
                     "the workers' epochs are their builds and their idle time"))
          (when (zero? w)
            (check-true (positive? (hash-ref (hash-ref r 'counts) 'tensors_per_sample))))))))

  (test-case "every pipeline name the cell takes is a loader the driver plans"
    (check-equal? (remove-duplicates (map (lambda (b) (hash-ref b 'pipeline)) (plan)))
                  '(cifar-png jpeg))))
