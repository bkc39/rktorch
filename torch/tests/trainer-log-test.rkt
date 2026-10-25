#lang racket/base

(module+ test
  (require (only-in racket/list append-map)
           rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" mean ones shape sum tensor)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../train/private/hooks.rkt"
                    callback-hook? definable-hooks hook-log hook-name)
           "../train.rkt")

  (struct collecting (metrics hparams finalized) #:mutable
    #:methods gen:logger
    [(define (logger-log-hyperparams! l params)
       (set-collecting-hparams! l (append (collecting-hparams l) (list params))))
     (define (logger-log-metrics! l metrics step)
       (set-collecting-metrics! l (append (collecting-metrics l) (list metrics))))
     (define (logger-finalize! l status)
       (set-collecting-finalized! l (append (collecting-finalized l) (list status))))])

  (define (collector) (collecting '() '() '()))

  (define (logged-keys l)
    (append-map hash-keys (collecting-metrics l)))

  (define labels (tensor '((1.0) (0.0) (1.0) (1.0) (0.0))))

  (define (label-loader #:batch-size [b 2])
    (dataloader (tensor-dataset (ones 5 1) labels) #:batch-size b))

  (define-trainer Logging (net reduce)
    (define-init (r) (set! net (Linear 1 1)) (set! reduce r))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (log! 'acc (mean y) #:on-step? #f #:on-epoch? #t #:batch-size (car (shape y)))
      (log! 'total (sum y) #:on-step? #f #:on-epoch? #t #:reduce reduce)
      (log! 'loss (mse-loss (net x) y) #:on-step? #t #:on-epoch? #t)
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (log! 'val/seen (sum y))
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
    (define l (collector))
    (define r (fit (Logging 'mean) (label-loader) #:val (label-loader) #:max-epochs 1
                   #:check-val-every-n-epoch 2 #:num-sanity-val-steps 2 #:logger l))
    (check-false (run-metric r 'val/seen))
    (check-false (memq 'val/seen (logged-keys l))))

  (test-case "L-L4: a key logged per step and per epoch reaches the logger twice"
    (define l (collector))
    (fit (Logging 'mean) (label-loader) #:max-epochs 1 #:log-every-n-steps 1 #:logger l)
    (check-not-false (memq 'loss-step (logged-keys l)))
    (check-not-false (memq 'loss-epoch (logged-keys l))))

  (test-case "L-L5: run-metric's default, and the last epoch's value after fit"
    (define r (fit (Logging 'sum) (label-loader) #:max-epochs 2 #:logger #f))
    (check-false (run-metric r 'never-logged))
    (check-equal? (run-metric r 'never-logged #:default 'none) 'none)
    (check-equal? (run-metric r 'total) 3.0)
    (check-equal? (hash-ref (run-metrics r) 'total) 3.0))

  (define-trainer Tuned (net lr)
    (define-init (rate) (set! net (Linear 1 1)) (set! lr rate))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "L-L6: the hparams reach the logger once"
    (define l (collector))
    (fit (Tuned 0.25) (label-loader) #:max-epochs 2 #:logger l)
    (check-equal? (length (collecting-hparams l)) 1)
    (check-equal? (hash-ref (car (collecting-hparams l)) 'lr) 0.25)
    (check-equal? (collecting-finalized l) '(success)))

  (define callback-hooks (filter callback-hook? definable-hooks))

  (define (logging-callback outcomes)
    (define kws
      (for/list ([h (in-list callback-hooks)])
        (string->keyword (symbol->string (hook-name h)))))
    (define procs
      (for/list ([h (in-list callback-hooks)])
        (lambda (r . _)
          (define ok?
            (with-handlers ([exn:fail:contract? (lambda (e) #f)])
              (run-log! r (hook-name h) 1.0)
              #t))
          (hash-set! outcomes (hook-name h) ok?)
          (and (eq? (hook-name h) 'state-dict) (hasheq)))))
    (define sorted (sort (map cons kws procs) keyword<? #:key car))
    (keyword-apply lambda-callback (map car sorted) (map cdr sorted) '()))

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
                    Configuring))))))
