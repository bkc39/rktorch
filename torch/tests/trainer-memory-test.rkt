#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in (submod "../data/workers.rkt" private) live-worker-count)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait!)
           (only-in "../foreign.rkt" cpu-device detach native-memory-use requires-grad?)
           (only-in "../main.rkt" ones sum zeros)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt"
                    collecting-hparams collecting-logger collecting-metrics)
           (only-in (submod "../train/logging.rkt" private) epoch-accumulators host-reads))

  (define (collect!)
    (define-values (seen? drained?) (collect-and-wait!))
    (check-true (and seen? drained?) "collection drained"))

  (define (cpu-bytes)
    (collect!)
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

  (define-callback late-logger ()
    #:kind monitor
    (define-hook (on-train-epoch-end) (log! 'late (sum (ones 2)))))

  (test-case "L-R1: a late callback's epoch-end log! shares the epoch's one read"
    (define before (host-reads))
    (define l (collecting-logger))
    (fit (Keyed '(loss)) (loader 14) #:max-epochs 2 #:log-every-n-steps 3 #:logger l
         #:callbacks (list (late-logger)))
    (check-equal? (- (host-reads) before) (+ (quotient 14 3) 2))
    (check-equal? (for/list ([m (in-list (collecting-metrics l))]
                             #:when (hash-has-key? m 'late))
                    (hash-ref m 'late))
                  '(2.0 2.0)))

  (define-trainer Holding (net)
    (define-init (data label) (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "a trainer keeps no init argument that is not readable"
    (define held (make-weak-box (zeros 1000 1000)))
    (define tr (Holding (weak-box-value held) "kept"))
    (collect!)
    (check-false (weak-box-value held))
    (define l (collecting-logger))
    (fit tr (loader 2 #:width 1) #:max-epochs 1 #:logger l)
    (check-equal? (collecting-hparams l) (list (hasheq 'label "kept"))))

  (define-trainer Wide (net mode weak kept)
    (define-init (m w)
      (set! net (Linear 1024 1024))
      (set! mode m)
      (set! weak w)
      (set! kept '()))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (define out (net x))
      (define loss (mse-loss out x))
      (define logged (detach loss))
      (log! 'loss logged #:on-step? #t #:on-epoch? #t)
      (log! 'raw loss #:on-step? #t #:on-epoch? #t)
      (set-box! weak (cons (list (make-weak-box loss) (make-weak-box logged)) (unbox weak)))
      (case mode
        [(retain) (set! kept (list* out loss logged kept))]
        [(drop) (ones 512 1024)]
        [else (void)])
      loss))

  (define (wide-fit mode #:weak [weak (box '())] #:callbacks [callbacks '()])
    (fit (Wide mode weak) (loader 640 #:width 1024 #:batch-size 64) #:max-steps 60
         #:logger #f #:callbacks callbacks))

  (define (mid-epoch-callback f)
    (lambda-callback
     #:on-train-batch-end
     (lambda (r loss extra . batch)
       (case (run-global-step r)
         [(55) (f r)]
         [else (void)]))))

  (define (earlier-survivors mode)
    (define weak (box '()))
    (define alive #f)
    (define census
      (mid-epoch-callback
       (lambda (r)
         (collect!)
         (match-define (cons _ earlier) (unbox weak))
         (set! alive
               (for*/sum ([step (in-list earlier)]
                          [wb (in-list step)]
                          #:when (weak-box-value wb))
                 1)))))
    (wide-fit mode #:weak weak #:callbacks (list census))
    alive)

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
    (wide-fit mode #:callbacks (list gauge))
    (- after-many after-one))

  (define growth-tolerance (* 1024 1024))

  (test-case "L-R2: no step's loss or logged value outlives its step"
    (check-equal? (earlier-survivors #f) 0))

  (test-case "L-R2's weak boxes see a trainer that keeps every step's loss"
    (check-equal? (earlier-survivors 'retain) (* 2 54)))

  (test-case "L-R2: log! accumulates a detached value when given the loss itself"
    (define grads #f)
    (define probe
      (mid-epoch-callback
       (lambda (r)
         (set! grads
               (for/list ([(key acc) (in-hash (epoch-accumulators r))])
                 (cons key (requires-grad? acc)))))))
    (wide-fit #f #:callbacks (list probe))
    (check-equal? (sort grads symbol<? #:key car) '((loss . #f) (raw . #f))))

  (test-case "L-R2: the ledger after many steps stays near the ledger after one"
    (check-= (ledger-growth #f) 0 growth-tolerance))

  (test-case "L-R2's gauge collects a 2 MiB tensor dropped at every step"
    (check-= (ledger-growth 'drop) 0 growth-tolerance))

  (test-case "L-R2's tolerance catches a trainer that keeps every step's output"
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
