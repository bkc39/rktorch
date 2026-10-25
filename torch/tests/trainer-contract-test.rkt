#lang racket/base

(module misbehaving racket/base
  (require (only-in "../main.rkt" ones tensor)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd step-lr)
           "../train.rkt")

  (provide NotOptimizer StrayScheduler TwoOptimizers BadLoss ThreeValues
           LoopSkipper OverLogger StepLogger)

  (define-trainer NotOptimizer (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers 5)
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer StrayScheduler (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers
      (define stray (sgd (parameters (Linear 1 1)) #:lr 0.1))
      (values (sgd (parameters net) #:lr 0.1) (step-lr stray #:step-size 1)))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer TwoOptimizers (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers
      (values (sgd (parameters net) #:lr 0.1) (sgd (parameters net) #:lr 0.2)))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer BadLoss (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (tensor 3)))

  (define-trainer ThreeValues (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (values (mse-loss (net x) y) 1 2)))

  (define-trainer LoopSkipper (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (skip-epoch! run)
      (mse-loss (net x) y)))

  (define-trainer OverLogger (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (log! 'wide (ones 2))
      (mse-loss (net x) y)))

  (define-trainer StepLogger (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-train-epoch-start) (log! 'epoch 1.0 #:on-step? #t))))

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (submod ".." misbehaving))

  (define ((blaming pattern) e)
    (and (exn:fail:contract? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))
  (define blames-this-test
    (blaming #rx"blaming: [(][^)]*trainer-contract-test\\.rkt test[)]"))
  (define blames-the-trainer
    (blaming #rx"blaming: [(][^)]*trainer-contract-test\\.rkt misbehaving[)]"))

  (define (ones-loader n #:columns [columns 2])
    (dataloader (apply tensor-dataset (for/list ([_ (in-range columns)]) (ones n 1)))
                #:batch-size 1))

  (define-trainer Good (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y)))

  (define-trainer Unvalidated (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Stepless (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1)))

  (define-trainer Unoptimized (net)
    (define-init () (set! net (Linear 1 1)))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Manual (net)
    #:automatic-optimization? #f
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Tuned (net lr)
    (define-init () (set! net (Linear 1 1)) (set! lr 0.1))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-callback keeper (count)
    (define-init () (set! count 0))
    (define-state-key "keeper")
    (define-hook (state-dict) (hasheq 'count count))
    (define-hook (load-state! state) (set! count (hash-ref state 'count))))

  (define bad-fit-keywords
    `((#:accumulate-grad-batches . 0)
      (#:callbacks . (5))
      (#:check-val-every-n-epoch . 0)
      (#:ckpt-path . 5)
      (#:device . cuda)
      (#:fast-dev-run . 0)
      (#:gradient-clip-algorithm . l2)
      (#:gradient-clip-val . 0)
      (#:limit-train-batches . 1.5)
      (#:limit-val-batches . -1)
      (#:log-every-n-steps . 0)
      (#:logger . stdout)
      (#:max-epochs . 0)
      (#:max-steps . 0)
      (#:num-sanity-val-steps . -1)
      (#:precision . 16)
      (#:val . 5)
      (#:val-check-interval . 1.5)))

  (define (fit-with kw v)
    (define given
      (sort (append (list (cons kw v))
                    (if (memq kw '(#:max-epochs)) '() '((#:max-epochs . 1)))
                    (if (memq kw '(#:logger)) '() '((#:logger . #f))))
            keyword<? #:key car))
    (keyword-apply fit (map car given) (map cdr given) (list (Good) (ones-loader 2))))

  (test-case "each fit keyword's contract blames the caller"
    (for ([kv (in-list bad-fit-keywords)])
      (check-exn (lambda (e) (and (blames-this-test e)
                                  (regexp-match? #rx"^fit: contract violation" (exn-message e))))
                 (lambda () (fit-with (car kv) (cdr kv)))
                 (format "~a" (car kv)))))

  (test-case "each test and predict keyword's contract blames the caller"
    (check-exn blames-this-test (lambda () (test (Good) (ones-loader 2) #:device 'cpu)))
    (check-exn blames-this-test (lambda () (test (Good) (ones-loader 2) #:logger 'stdout)))
    (check-exn blames-this-test (lambda () (test (Good) (ones-loader 2) #:callbacks '(1))))
    (check-exn blames-this-test
               (lambda () (test (Good) (ones-loader 2) #:limit-test-batches 2.0)))
    (check-exn blames-this-test (lambda () (test (Good) (ones-loader 2) #:ckpt-path 1)))
    (check-exn blames-this-test (lambda () (predict (Good) (ones-loader 2) #:device 'cpu)))
    (check-exn blames-this-test
               (lambda () (predict (Good) (ones-loader 2) #:limit-predict-batches -1)))
    (check-exn blames-this-test (lambda () (predict 'trainer (ones-loader 2))))
    (check-exn blames-this-test (lambda () (fit (Good) 'loader #:max-epochs 1))))

  (define ((pre-violation message) e)
    (and (blames-this-test e)
         (regexp-match? (regexp-quote message) (exn-message e))))

  (test-case "each #:pre/name of fit, test and predict blames the caller"
    (check-exn (pre-violation "fit needs #:max-epochs or #:max-steps")
               (lambda () (fit (Good) (ones-loader 2) #:logger #f)))
    (check-exn (pre-violation "the trainer has no define-training-step")
               (lambda () (fit (Stepless) (ones-loader 2) #:max-epochs 1 #:logger #f)))
    (check-exn (pre-violation "automatic optimization needs define-optimizers")
               (lambda () (fit (Unoptimized) (ones-loader 2) #:max-epochs 1 #:logger #f)))
    (check-exn (pre-violation "a #:val loader needs a define-validation-step")
               (lambda () (fit (Unvalidated) (ones-loader 2) #:val (ones-loader 2)
                               #:max-epochs 1 #:logger #f)))
    (check-exn (pre-violation "#:limit-train-batches selects no batch from this loader")
               (lambda () (fit (Good) (ones-loader 2) #:limit-train-batches 0
                               #:max-epochs 1 #:logger #f)))
    (check-exn (pre-violation "#:limit-val-batches selects no batch from this loader")
               (lambda () (fit (Good) (ones-loader 2) #:val (ones-loader 2)
                               #:limit-val-batches 0.1 #:max-epochs 1 #:logger #f)))
    (check-exn (pre-violation "an integer #:val-check-interval exceeds the loader's batches")
               (lambda () (fit (Good) (ones-loader 2) #:val-check-interval 3
                               #:max-epochs 1 #:logger #f)))
    (check-exn (pre-violation
                "manual optimization takes no #:accumulate-grad-batches above 1 and no #:gradient-clip-val")
               (lambda () (fit (Manual) (ones-loader 2) #:accumulate-grad-batches 2
                               #:max-epochs 1 #:logger #f)))
    (check-exn (pre-violation "the trainer has no define-test-step")
               (lambda () (test (Good) (ones-loader 2) #:logger #f)))
    (check-exn (pre-violation "#:limit-test-batches selects no batch from this loader")
               (lambda () (test (Good) (ones-loader 2) #:limit-test-batches 0 #:logger #f)))
    (check-exn (pre-violation "the trainer has no define-predict-step")
               (lambda () (predict (Good) (ones-loader 2)))))

  (test-case "what a trainer's own bodies return blames the module defining it"
    (for ([make (list NotOptimizer StrayScheduler TwoOptimizers BadLoss ThreeValues)])
      (check-exn blames-the-trainer
                 (lambda () (fit (make) (ones-loader 2) #:max-epochs 1 #:logger #f))
                 (format "~a" make))))

  (test-case "a batch of the wrong arity names the step"
    (check-exn (lambda (e) (and (exn:fail:contract:arity? e)
                                (regexp-match? #rx"training-step" (exn-message e))))
               (lambda () (fit (Good) (ones-loader 2 #:columns 3) #:max-epochs 1
                               #:logger #f))))

  (test-case "log! takes a scalar, and only where its hook logs"
    (check-exn blames-the-trainer
               (lambda () (fit (OverLogger) (ones-loader 2) #:max-epochs 1 #:logger #f)))
    (check-exn (lambda (e) (and (blames-the-trainer e)
                                (regexp-match? #rx"this hook logs only per epoch"
                                               (exn-message e))))
               (lambda () (fit (StepLogger) (ones-loader 2) #:max-epochs 1 #:logger #f))))

  (test-case "loop control needs a live run in the right hook"
    (define r (fit (Good) (ones-loader 2) #:max-epochs 1 #:logger #f))
    (check-exn blames-this-test (lambda () (request-stop! r)))
    (check-exn blames-this-test (lambda () (skip-step! r)))
    (check-exn blames-this-test (lambda () (skip-epoch! r)))
    (check-exn blames-the-trainer
               (lambda () (fit (LoopSkipper) (ones-loader 2) #:max-epochs 1 #:logger #f))))

  (test-case "set-hparam! takes a define-hparams name"
    (check-exn blames-this-test (lambda () (set-hparam! (Tuned) 'momentum 0.9))))

  (test-case "two stateful callbacks may not share a state key"
    (check-exn exn:fail:contract?
               (lambda () (fit (Good) (ones-loader 2) #:max-epochs 1 #:logger #f
                               #:callbacks (list (keeper) (keeper)))))))
