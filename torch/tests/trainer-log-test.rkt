#lang racket/base

(module+ test
  (require (only-in racket/list append-map)
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
                    collecting-logger collecting-metrics))

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
      (mse-loss (net x) y)))

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

  (test-case "L-L7: log! is allowed exactly in the hooks the table allows"
    (define outcomes (make-hasheq))
    (define cb (logging-callback outcomes))
    (define tr (Logging 'mean))
    (fit tr (label-loader) #:val (label-loader) #:max-epochs 1 #:logger #f
         #:callbacks (list cb))
    (test tr (label-loader) #:logger #f #:callbacks (list cb))
    (for ([h (in-list callback-hooks)]
          #:when (hash-has-key? outcomes (hook-name h)))
      (check-equal? (hash-ref outcomes (hook-name h)) (and (hook-log h) #t)
                    (format "~a" (hook-name h)))))

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
