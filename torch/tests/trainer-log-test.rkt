#lang racket/base

(module+ test
  (require (only-in racket/list append-map make-list partition remove-duplicates)
           (only-in racket/match match)
           rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" mean ones shape sum tensor)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../train/private/hooks.rkt"
                    callback-hook? definable-hooks hook-log hook-name)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt"
                    apply-keywords collecting-finalized collecting-hparams
                    collecting-logger collecting-metrics collecting-steps))

  (define (logged-keys l)
    (append-map hash-keys (collecting-metrics l)))

  (define labels (tensor '((1.0) (0.0) (1.0) (1.0) (0.0))))

  (define (label-loader #:batch-size [b 2])
    (dataloader (tensor-dataset (ones 5 1) labels) #:batch-size b))

  (define-trainer Logging (net reduce)
    (define-init (r) (set! net (Linear 1 1)) (set! reduce r))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (log! 'acc (mean y) #:on-step? #f #:on-epoch? #t
            #:batch-size (match (shape y) [(cons rows _) rows]))
      (log! 'total (sum y) #:on-step? #f #:on-epoch? #t #:reduce reduce)
      (log! 'loss (mse-loss (net x) y) #:on-step? #t #:on-epoch? #t)
      (log! 'epoch (run-epoch run) #:on-step? #f #:on-epoch? #t #:reduce 'max)
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (log! 'val/seen (sum y))
      (mse-loss (net x) y))
    (define-test-step (x y)
      (mse-loss (net x) y))
    (define-predict-step (x y)
      (net x)))

  (test-case "L-L1: an epoch mean is weighted by batch size"
    (define r (fit (Logging 'mean) (label-loader) #:max-epochs 1 #:logger #f))
    (check-= (run-metric r 'acc) 0.6 1e-6))

  (test-case "L-L2: sum, max and min reduce exactly"
    (for ([reduce (in-list '(sum max min))]
          [expected (in-list '(3.0 2.0 0.0))])
      (define r (fit (Logging reduce) (label-loader) #:max-epochs 1 #:logger #f))
      (check-equal? (run-metric r 'total) expected (format "~a" reduce))))

  (test-case "L-L3: the sanity check's metrics are discarded"
    (define l (collecting-logger))
    (define sanity-batches 0)
    (define counter
      (lambda-callback
       #:on-validation-batch-end
       (lambda (r out . batch)
         (when (eq? (run-stage r) 'sanity-check)
           (set! sanity-batches (add1 sanity-batches))))))
    (define r (fit (Logging 'mean) (label-loader) #:val (label-loader) #:max-epochs 1
                   #:check-val-every-n-epoch 2 #:num-sanity-val-steps 2 #:logger l
                   #:callbacks (list counter)))
    (check-equal? sanity-batches 2)
    (check-false (run-metric r 'val/seen))
    (check-false (memq 'val/seen (logged-keys l))))

  (test-case "L-L4: a key logged per step and per epoch reaches the logger twice"
    (define l (collecting-logger))
    (fit (Logging 'mean) (label-loader) #:max-epochs 1 #:log-every-n-steps 1 #:logger l)
    (check-not-false (memq 'loss-step (logged-keys l)))
    (check-not-false (memq 'loss-epoch (logged-keys l))))

  (test-case "L-L5: run-metric's default, and the last epoch's value after fit"
    (define r (fit (Logging 'sum) (label-loader) #:max-epochs 2 #:logger #f))
    (check-false (run-metric r 'never-logged))
    (check-equal? (run-metric r 'never-logged #:default 'none) 'none)
    (check-equal? (run-metric r 'total) 3.0)
    (check-= (run-metric r 'epoch) 1 0)
    (check-= (hash-ref (run-metrics r) 'epoch) 1 0))

  (define-trainer Tuned (net lr)
    (define-init (rate) (set! net (Linear 1 1)) (set! lr rate))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "L-L6: the hparams reach the logger once"
    (define l (collecting-logger))
    (fit (Tuned 0.25) (label-loader) #:max-epochs 2 #:logger l)
    (check-match (collecting-hparams l) (list (hash-table ('lr 0.25) _ ...)))
    (check-equal? (collecting-finalized l) '(success)))

  (define callback-hooks (filter callback-hook? definable-hooks))

  (define ((logging-hook outcomes name) r . _)
    (define ok?
      (with-handlers ([exn:fail:contract? (lambda (e) #f)])
        (run-log! r name 1.0)
        #t))
    (hash-set! outcomes name ok?)
    (and (eq? name 'state-dict) (hasheq)))

  (define (logging-callback outcomes)
    (apply-keywords lambda-callback
                    (for/list ([h (in-list callback-hooks)])
                      (cons (string->keyword (symbol->string (hook-name h)))
                            (logging-hook outcomes (hook-name h))))))

  (define checkpoint-hooks '(state-dict load-state! on-save-checkpoint on-load-checkpoint))

  (test-case "L-L7: log! is allowed exactly in the hooks the table allows"
    (define outcomes (make-hasheq))
    (define cb (logging-callback outcomes))
    (define tr (Logging 'mean))
    (fit tr (label-loader) #:val (label-loader) #:max-epochs 1 #:logger #f
         #:callbacks (list cb))
    (test tr (label-loader) #:logger #f #:callbacks (list cb))
    (predict tr (label-loader) #:callbacks (list cb))
    (define failing (lambda-callback #:on-train-end (lambda (r) (error 'on-train-end "planned"))))
    (check-exn #rx"planned"
               (lambda ()
                 (fit tr (label-loader) #:max-epochs 1 #:logger #f
                      #:callbacks (list cb failing))))
    (check-equal? (for/list ([h (in-list callback-hooks)]
                             #:unless (hash-has-key? outcomes (hook-name h)))
                    (hook-name h))
                  checkpoint-hooks)
    (for ([h (in-list callback-hooks)]
          #:when (hash-has-key? outcomes (hook-name h)))
      (check-equal? (hash-ref outcomes (hook-name h)) (and (hook-log h) #t)
                    (format "~a" (hook-name h)))))

  (define-trainer Stepping (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (log! 'loss 1.0 #:on-step? #t #:on-epoch? #f)
      (log! 'train-only 1.0 #:on-step? #t #:on-epoch? #f)
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (log! 'loss 100.0 #:on-step? #t #:on-epoch? #f)
      (log! 'val-only 2.0 #:on-step? #t #:on-epoch? #f)
      (mse-loss (net x) y))
    (define-test-step (x y)
      (log! 'test-loss 3.0 #:on-step? #t #:on-epoch? #f)
      (mse-loss (net x) y)))

  (test-case "a per-step log in test-step reaches the logger at each test batch"
    (define l (collecting-logger))
    (test (Stepping) (label-loader) #:logger l)
    (check-equal? (collecting-steps l)
                  (for/list ([step (in-range 3)]) (cons step (hasheq 'test-loss 3.0)))))

  (define-trainer Labelled (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (log! 'tl 1.0 #:on-step? #t #:on-epoch? #t)
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (log! 'vl 2.0 #:on-step? #t #:on-epoch? #t)
      (mse-loss (net x) y))
    (define-test-step (x y)
      (log! 'xl 3.0 #:on-step? #t #:on-epoch? #t)
      (mse-loss (net x) y)))

  (define (logged-steps l) (map car (collecting-steps l)))

  (test-case "rows carry Lightning's step labels: the steps taken before the current one"
    (define l (collecting-logger))
    (fit (Labelled) (label-loader) #:val (label-loader #:batch-size 3) #:max-epochs 2
         #:log-every-n-steps 1 #:num-sanity-val-steps 0 #:logger l)
    (check-equal? (logged-steps l) '(0 1 2 0 1 2 2 3 4 5 2 3 5 5))
    (define t (collecting-logger))
    (test (Labelled) (label-loader) #:logger t)
    (test (Labelled) (label-loader #:batch-size 3) #:logger t)
    (check-equal? (logged-steps t) '(0 1 2 0 0 1 0)))

  (test-case "a validation inside an accumulation window is labelled with the steps taken"
    (define l (collecting-logger))
    (fit (Labelled) (label-loader) #:val (label-loader #:batch-size 5) #:max-epochs 1
         #:accumulate-grad-batches 2 #:val-check-interval 1
         #:log-every-n-steps 1 #:num-sanity-val-steps 0 #:logger l)
    (check-equal? (logged-steps l) '(0 0 0 1 0 1 2 1 1)))

  (define-trainer OwnEpoch (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (log! 'epoch 7.0 #:on-step? #f #:on-epoch? #t)
      (mse-loss (net x) y)))

  (define (epoch-tags l)
    (for/list ([row (in-list (collecting-steps l))])
      (match row
        [(cons step metrics) (list step (hash-ref metrics 'epoch #f))])))

  (test-case "every row but a validation or test step's carries Lightning's epoch key"
    (define l (collecting-logger))
    (fit (Labelled) (label-loader) #:val (label-loader #:batch-size 3) #:max-epochs 2
         #:log-every-n-steps 1 #:num-sanity-val-steps 0 #:logger l)
    (check-equal? (epoch-tags l)
                  '((0 0) (1 0) (2 0) (0 #f) (1 #f) (2 0) (2 0)
                    (3 1) (4 1) (5 1) (2 #f) (3 #f) (5 1) (5 1)))
    (define t (collecting-logger))
    (test (Labelled) (label-loader) #:logger t)
    (check-equal? (epoch-tags t) '((0 #f) (1 #f) (2 #f) (0 0)))
    (define own (collecting-logger))
    (fit (OwnEpoch) (label-loader) #:max-epochs 2 #:logger own)
    (check-equal? (map (lambda (m) (hash-ref m 'epoch)) (collecting-metrics own)) '(7.0 7.0)))

  (test-case "an epoch with no step yet labels its training row 0"
    (define l (collecting-logger))
    (define skip-first
      (lambda-callback #:on-before-optimizer-step
                       (lambda (r opt) (when (zero? (run-epoch r)) (skip-step! r)))))
    (fit (Labelled) (label-loader) #:max-epochs 2 #:log-every-n-steps 1 #:logger l
         #:callbacks (list skip-first))
    (check-equal? (logged-steps l) '(0 0 1 2 2)))

  (test-case "request-stop! publishes the step's row off the #:log-every-n-steps cadence"
    (define l (collecting-logger))
    (define stopper
      (lambda-callback #:on-train-batch-end
                       (lambda (r loss extra . batch)
                         (when (= (run-global-step r) 3) (request-stop! r)))))
    (fit (Labelled) (dataloader (tensor-dataset (ones 10 1) (ones 10 1)) #:batch-size 1)
         #:max-epochs 5 #:log-every-n-steps 50
         #:logger l #:callbacks (list stopper))
    (check-equal? (collecting-steps l)
                  (list (cons 2 (hasheq 'tl-step 1.0 'epoch 0)) (cons 2 (hasheq 'tl-epoch 1.0 'epoch 0)))))

  (test-case "a per-step log in validation-step stays out of the training reads"
    (define l (collecting-logger))
    (fit (Stepping) (label-loader) #:val (label-loader) #:max-epochs 2
         #:log-every-n-steps 1 #:num-sanity-val-steps 0 #:logger l)
    (define-values (train-reads val-reads)
      (partition (lambda (m) (hash-has-key? m 'train-only)) (collecting-metrics l)))
    (check-equal? (length train-reads) 6)
    (check-equal? (remove-duplicates train-reads)
                  (list (hasheq 'loss 1.0 'train-only 1.0 'epoch 0)
                        (hasheq 'loss 1.0 'train-only 1.0 'epoch 1)))
    (check-equal? val-reads (make-list 6 (hasheq 'loss 100.0 'val-only 2.0))))

  (test-case "run-metric inside an epoch answers the latest step value"
    (define seen '())
    (define (watch r . _) (set! seen (cons (run-metric r 'loss) seen)))
    (fit (Stepping) (label-loader) #:max-epochs 1 #:log-every-n-steps 1 #:logger #f
         #:callbacks (list (lambda-callback #:on-train-batch-start watch
                                            #:on-train-batch-end watch)))
    (check-equal? (reverse seen) '(#f 1.0 1.0 1.0 1.0 1.0)))

  (define-trainer EpochTagged (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (log! 'e (run-epoch run) #:on-epoch? #t)
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (log! 'v (run-epoch run))
      (mse-loss (net x) y)))

  (test-case "run-metric in an epoch-end hook answers the epoch being closed"
    (define seen '())
    (define ((watch name key) r)
      (set! seen (cons (list name (run-epoch r) (run-metric r key)) seen)))
    (define cb (lambda-callback #:on-validation-epoch-end (watch 'validation 'v)
                                #:on-train-epoch-end (watch 'train 'e)))
    (fit (EpochTagged) (label-loader) #:val (label-loader) #:max-epochs 2
         #:num-sanity-val-steps 0 #:logger #f #:callbacks (list cb))
    (check-equal? (reverse seen)
                  '((validation 0 0.0) (train 0 0.0) (validation 1 1.0) (train 1 1.0))))

  (test-case "E7: a lexical log! in a hook that cannot log"
    (check-exn #rx"log!: on-fit-start cannot log"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-trainer Early ()
                      (define-hook (on-fit-start) (log! 'x 1.0)))
                    Early))))
    (check-exn #rx"log!: configure-optimizers cannot log"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-trainer Configuring (net)
                      (define-init () (set! net (Linear 1 1)))
                      (define-optimizers (log! 'x 1.0) (sgd (parameters net) #:lr 0.1)))
                    Configuring))))
    (check-exn #rx"log!: configure-callbacks cannot log"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-trainer Wiring ()
                      (define-callbacks (log! 'x 1.0) '()))
                    Wiring))))))
