#lang racket/base

(module+ test
  (require (only-in racket/list last take)
           rackunit
           (only-in "summary-m0.rkt"
                    arm-rate block-ratios cell-row done-when k6-column m0-markdown scaling
                    training-row valid-blocks))

  (define (arm pipeline workers which rate
               #:block block #:round [round 0] #:racket [racket "9.3"] #:load [load 0.5]
               #:error [error #f] #:gc [gc 10])
    (define train? (equal? pipeline "train"))
    (for/fold ([r (hasheq 'kind "arm" 'run_id "run" 'round round 'block block
                          'pipeline pipeline 'workers workers 'arm which 'racket racket
                          'max_load 2.0 'load_outside load 'gc_ms gc 'gc_count 3
                          'samples 1000 'os_threads_baseline 10
                          'os_threads_peak 10 'os_threads_idle 12
                          'os_threads_active (+ 12 (* 2 workers)) 'worker_threads '(2 3))])
              ([kv (in-list (list (cons (if train? 'main_steps_per_s 'samples_per_s) rate)
                                  (cons 'ops_per_s (and train? 1000.0))
                                  (cons 'error error)))]
               #:when (cdr kv))
      (hash-set r (car kv) (cdr kv))))

  (define (keyword-block pipeline workers b rk py [pyp #f]
                         #:round [round 0] #:racket [racket "9.3"] #:load [load 0.5]
                         #:error [error #f])
    (define (one which rate)
      (arm pipeline workers which rate #:block b #:round round #:racket racket #:load load
           #:error (and (equal? which "rktorch") error)))
    (append (list (one "rktorch" rk) (one "pytorch" py))
            (if pyp (list (one "pytorch-persistent" pyp)) '())))

  (test-case "a ratio is the median over blocks of rktorch over its own block's PyTorch arm"
    (define records (append (keyword-block "jpeg" 4 0 100.0 200.0 150.0)
                            (keyword-block "jpeg" 4 1 #:round 1 200.0 400.0 300.0)
                            (keyword-block "jpeg" 4 2 #:round 2 300.0 600.0 450.0)))
    (check-equal? (block-ratios records "jpeg" 4 "pytorch") '(0.5 0.5 0.5)
                  "a 3x drift between blocks leaves the ratio alone")
    (check-equal? (map (lambda (x) (/ (round (* 100 x)) 100.0))
                       (block-ratios records "jpeg" 4 "pytorch-persistent"))
                  '(0.67 0.67 0.67))
    (check-equal? (arm-rate records "jpeg" 4 "rktorch") 200.0)
    (define row (cell-row records "jpeg" 4))
    (check-equal? (hash-ref row 'vs-pytorch) 0.5)
    (check-equal? (hash-ref row 'blocks) 3)
    (check-equal? (hash-ref row 'rk-threads) 2.0 "the worker traversal's OS threads over the worker-free one's, per worker")
    (check-equal? (hash-ref row 'py-threads) 3 "the busiest worker process"))

  (test-case "a block with an error or an outside load at the limit is dropped and counted"
    (define records (append (keyword-block "cifar" 0 0 100.0 200.0)
                            (keyword-block "cifar" 0 1 #:round 1 10.0 200.0 #:load 2.0)
                            (keyword-block "cifar" 0 2 #:round 2 10.0 200.0 #:error "boom")))
    (define-values (valid dropped) (valid-blocks records "cifar" 0))
    (check-equal? (list (length valid) dropped) '(1 2))
    (check-equal? (block-ratios records "cifar" 0 "pytorch") '(0.5))
    (check-equal? (hash-ref (cell-row records "cifar" 0) 'dropped) 2))

  (test-case "a block missing an arm its cell runs is dropped, so an interrupted run's rate is not used"
    (define records (append (keyword-block "jpeg" 4 0 100.0 200.0 150.0)
                            (keyword-block "jpeg" 4 1 #:round 1 900.0 400.0)
                            (take (keyword-block "jpeg" 0 2 #:round 1 900.0 400.0) 1)))
    (define-values (valid dropped) (valid-blocks records "jpeg" 4))
    (check-equal? (list (length valid) dropped) '(1 1))
    (check-equal? (arm-rate records "jpeg" 4 "rktorch") 100.0)
    (check-equal? (hash-ref (cell-row records "jpeg" 0) 'dropped) 1)
    (check-false (arm-rate records "jpeg" 0 "rktorch")))

  (test-case "K6 reads the rktorch arms of accepted blocks only"
    (define clean (keyword-block "jpeg" 0 0 100.0 200.0))
    (define py-failed (for/list ([r (in-list (keyword-block "jpeg" 0 1 #:round 1 100.0 200.0))])
                        (if (equal? (hash-ref r 'arm) "pytorch") (hash-set r 'error "boom") r)))
    (check-equal? (hash-count (k6-column py-failed "jpeg")) 0)
    (check-equal? (k6-column (append clean py-failed) "jpeg") (k6-column clean "jpeg")))

  (test-case "scaling is w4 over w0 on each side; the persistent twin's base is the plain twin's w0"
    (define records (append (keyword-block "cifar-png" 0 0 100.0 400.0)
                            (keyword-block "cifar-png" 4 1 250.0 1000.0 2000.0)))
    (check-equal? (scaling records "cifar-png" "rktorch") 2.5)
    (check-equal? (scaling records "cifar-png" "pytorch") 2.5)
    (check-equal? (scaling records "cifar-png" "pytorch-persistent") 5.0)
    (check-false (scaling records "jpeg" "rktorch")))

  (test-case "the training arm compares main's steps per second within each block"
    (define records (append (keyword-block "train" 4 0 100.0 400.0)
                            (keyword-block "train" 4 1 #:round 1 300.0 600.0)))
    (define t (training-row records 4))
    (check-equal? (hash-ref t 'vs-pytorch) 0.375)
    (check-equal? (hash-ref t 'rk-ops) 1000.0))

  (define (op-row op threads us #:rep [rep 0])
    (hasheq 'kind "op" 'impl "rktorch" 'op op 'layer "facade" 'threads threads 'rep rep
            'us_per_op us))

  (test-case "the done-when rows mark each target met, unmet or n/a"
    (define records
      (append (keyword-block "cifar-png" 0 0 100.0 100.0)
              (keyword-block "cifar-png" 4 1 400.0 500.0 500.0)
              (keyword-block "jpeg" 0 2 100.0 100.0)
              (keyword-block "jpeg" 4 3 200.0 500.0 600.0)
              (keyword-block "train" 4 4 400.0 300.0)))
    (define op-cost (list (op-row "add-8x8" 1 2.0) (op-row "add-8x8" 8 5.0)
                          (op-row "zeros-8x8" 1 2.0) (op-row "zeros-8x8" 8 3.0)))
    (define rows (done-when records op-cost))
    (define (verdict target cell)
      (for/first ([r (in-list rows)]
                  #:when (and (regexp-match? (regexp-quote target) (car r))
                              (regexp-match? (regexp-quote cell) (cadr r))))
        (cadddr r)))
    (check-equal? (verdict "w4 ≥ 3× w0" "cifar-png") "met")
    (check-equal? (verdict "w4 ≥ 3× w0" "jpeg") "unmet")
    (check-equal? (verdict "w4 ≥ 3× w0" "cifar-batched") "n/a")
    (check-equal? (verdict "rk ÷ py ≥ 0.5" "cifar-png") "met")
    (check-equal? (verdict "rk ÷ py ≥ 0.5" "jpeg") "unmet")
    (check-equal? (verdict "training rate" "train") "met")
    (check-equal? (verdict "K3" "add-8x8") "unmet" "the worst op decides K3")
    (check-equal? (cadddr (last (done-when records '()))) "n/a" "no op-cost records, no K3"))

  (test-case "the summary splits by Racket version, and compares them when there are two"
    (define one (append (keyword-block "jpeg" 0 0 100.0 200.0)
                        (keyword-block "train" 1 1 100.0 200.0)))
    (define md (m0-markdown one))
    (check-regexp-match #rx"## Racket 9.3" md)
    (check-false (regexp-match? #rx"by Racket version" md))
    (define both (append one (keyword-block "jpeg" 0 0 #:racket "9.1" 150.0 200.0)))
    (define split (m0-markdown both))
    (check-regexp-match #rx"## Racket 9.1" split)
    (check-regexp-match #rx"## rk ÷ py by Racket version" split)
    (check-regexp-match #rx"\\| jpeg \\| 0 \\| 0.50 \\| 0.75 \\|" split)))
