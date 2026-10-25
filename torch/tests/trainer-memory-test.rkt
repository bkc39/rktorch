#lang racket/base

(module+ test
  (require (only-in racket/math exact-ceiling)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../foreign.rkt" cpu-device native-memory-use)
           (only-in "../main.rkt" ones sum)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in (submod "../train/logging.rkt" private) host-reads))

  (define (cpu-bytes)
    (collect-garbage)
    (collect-garbage)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (loader n #:workers [workers 0])
    (dataloader (tensor-dataset (ones n 4) (ones n 1)) #:batch-size 2 #:workers workers))

  (define-trainer Keyed (net keys fault)
    (define-init (k [f #f]) (set! net (Linear 4 1)) (set! keys k) (set! fault f))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (for ([key (in-list keys)])
        (log! key loss #:on-step? #t #:on-epoch? #t))
      (log! 'total (sum y) #:on-step? #f #:on-epoch? #t #:reduce 'sum)
      (case (and (= (run-batch-index run) 1) fault)
        [(skip) (skip-epoch! run)]
        [(stop) (request-stop! run)]
        [(fail) (error 'training-step "planned failure")]
        [else (void)])
      loss))

  (define (reads-per-fit keys #:epochs epochs #:batches batches #:every every)
    (define before (host-reads))
    (fit (Keyed keys) (loader (* 2 batches)) #:max-epochs epochs
         #:log-every-n-steps every #:logger #f)
    (- (host-reads) before))

  (test-case "L-R1: log! reads the host once per logged step and once per epoch"
    (for* ([keys (in-list '((loss) (loss a b c d)))]
           [every (in-list '(1 3 50))])
      (check-equal? (reads-per-fit keys #:epochs 2 #:batches 7 #:every every)
                    (* 2 (add1 (exact-ceiling (/ 7 every))))
                    (format "~a keys, every ~a" (length keys) every))))

  (test-case "L-R2: the ledger after many steps stays near the ledger after one"
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
    (fit (Keyed '(loss a b)) (loader 40) #:max-steps 60 #:logger #f #:callbacks (list gauge))
    (check-true (< (abs (- after-many after-one)) (* 1024 1024))
                (format "after 1 step ~a bytes, after 60 ~a" after-one after-many)))

  (test-case "L-R3: two fresh fits leave the ledger at the same baseline"
    (define (one-fit)
      (fit (Keyed '(loss)) (loader 20) #:max-epochs 2 #:logger #f)
      (cpu-bytes))
    (one-fit)
    (define first (one-fit))
    (define second (one-fit))
    (check-true (< (abs (- second first)) (* 64 1024))
                (format "first ~a bytes, second ~a" first second)))

  (define (live-threads thunk)
    (define c (make-custodian))
    (with-handlers ([exn:fail? void])
      (parameterize ([current-custodian c]) (thunk)))
    (for/list ([t (in-list (custodian-managed-list c (current-custodian)))]
               #:when (and (thread? t) (not (sync/timeout 5 (thread-dead-evt t)))))
      t))

  (test-case "L-R4: an early exit finishes the traversal and its workers"
    (for ([fault (in-list '(skip stop fail #f))])
      (check-equal?
       (live-threads
        (lambda ()
          (fit (Keyed '(loss) fault) (loader 12 #:workers 2) #:max-epochs 1 #:logger #f
               #:limit-train-batches (if fault 1.0 2))))
       '()
       (format "~a" (or fault 'limit))))))
