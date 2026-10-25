#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in (submod "../data/workers.rkt" private) live-worker-count)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait!)
           (only-in "../foreign.rkt" cpu-device native-memory-use)
           (only-in "../main.rkt" ones sum)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in (submod "../train/logging.rkt" private) host-reads))

  (define (cpu-bytes)
    (collect-and-wait!)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (loader n #:workers [workers 0] #:width [width 4] #:batch-size [b 2])
    (dataloader (tensor-dataset (ones n width) (ones n 1)) #:batch-size b #:workers workers))

  (define-trainer Keyed (net keys fault)
    (define-init (k [f #f]) (set! net (Linear 4 1)) (set! keys k) (set! fault f))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (for ([key (in-list keys)])
        (log! key loss #:on-step? #t #:on-epoch? #t))
      (log! 'total (sum y) #:on-step? #f #:on-epoch? #t #:reduce 'sum)
      (case (and (= (run-batch-index run) 1) fault)
        [(stop) (request-stop! run)]
        [(fail) (error 'training-step "planned failure")]
        [else (void)])
      loss)
    (define-hook (on-train-batch-start . batch)
      (when (and (eq? fault 'skip) (= (run-batch-index run) 1))
        (skip-epoch! run))))

  (define (reads-per-fit keys #:epochs epochs #:batches batches #:every every)
    (define before (host-reads))
    (fit (Keyed keys) (loader (* 2 batches)) #:max-epochs epochs
         #:log-every-n-steps every #:logger #f)
    (- (host-reads) before))

  (test-case "L-R1: log! reads the host at every n-th step of the fit and at each epoch end"
    (for* ([keys (in-list '((loss) (loss a b c d)))]
           [every (in-list '(1 3 50))])
      (check-equal? (reads-per-fit keys #:epochs 2 #:batches 7 #:every every)
                    (+ (quotient (* 2 7) every) 2)
                    (format "~a keys, every ~a" (length keys) every))))

  (define-trainer Wide (net mode kept)
    (define-init (m)
      (set! net (Linear 1024 1024))
      (set! mode m)
      (set! kept (box '())))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (define loss (mse-loss (net x) x))
      (log! 'loss loss #:on-step? #t #:on-epoch? #t)
      (case mode
        [(retain) (set-box! kept (cons loss (unbox kept)))]
        [(drop) (ones 512 1024)]
        [else (void)])
      loss))

  (define (ledger-growth mode)
    (define after-one #f)
    (define after-many #f)
    (define gauge
      (lambda-callback
       #:on-train-batch-end
       (lambda (r loss extra . batch)
         (case (run-global-step r)
           [(1) (set! after-one (cpu-bytes))]
           [(60) (set! after-many (cpu-bytes))]
           [else (void)]))))
    (fit (Wide mode) (loader 640 #:width 1024 #:batch-size 64) #:max-steps 60
         #:logger #f #:callbacks (list gauge))
    (- after-many after-one))

  (define growth-tolerance (* 1024 1024))

  (test-case "L-R2: the ledger after many steps stays near the ledger after one"
    (check-= (ledger-growth #f) 0 growth-tolerance))

  (test-case "L-R2's gauge collects a 2 MiB tensor dropped at every step"
    (check-= (ledger-growth 'drop) 0 growth-tolerance))

  (test-case "L-R2's tolerance catches a trainer that keeps every step's graph"
    (check-true (> (ledger-growth 'retain) growth-tolerance)))

  (test-case "L-R3: two fresh fits leave the ledger at the same baseline"
    (define (one-fit)
      (fit (Keyed '(loss)) (loader 20) #:max-epochs 2 #:logger #f)
      (cpu-bytes))
    (one-fit)
    (define first (one-fit))
    (define second (one-fit))
    (check-= second first (* 64 1024)))

  (test-case "L-R4: an early exit finishes the traversal and its workers"
    (for ([fault (in-list '(skip stop fail #f))])
      (define (run-fit)
        (fit (Keyed '(loss) fault) (loader 12 #:workers 2) #:max-epochs 1 #:logger #f
             #:limit-train-batches (if fault 1.0 2)))
      (cond
        [(eq? fault 'fail) (check-exn #rx"planned failure" run-fit)]
        [else (run-fit)])
      (check-equal? (live-worker-count #:wait 5) 0 (format "~a" (or fault 'limit))))))
