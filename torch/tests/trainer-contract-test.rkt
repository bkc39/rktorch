#lang racket/base

(module misbehaving racket/base
  (require (only-in "../main.rkt" ones tensor)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd step-lr)
           "../train.rkt")

  (provide NotOptimizer StrayScheduler TwoOptimizers BadLoss ThreeValues
           LoopSkipper OverLogger StepLogger EpochOffLogger)

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
    (define-hook (on-train-epoch-start) (log! 'epoch 1.0 #:on-step? #t)))

  (define-trainer EpochOffLogger (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-train-epoch-start) (log! 'epoch 1.0 #:on-epoch? #f))))

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (submod ".." misbehaving)
           (only-in "private/trainer-kit.rkt" apply-keywords interval-message))

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

  (define-trainer Complete (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y))
    (define-test-step (x y) (mse-loss (net x) y))
    (define-predict-step (x y) (net x)))

  (define (fit-quiet tr loader . keyword-values)
    (apply-keywords fit (append keyword-values '((#:max-epochs . 1) (#:logger . #f)))
                    tr loader))
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
    '((#:accumulate-grad-batches . 0)
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

  (define (override kv defaults)
    (cons kv (remove (car kv) defaults (lambda (k d) (eq? k (car d))))))

  (define ((violates who) e)
    (and (blames-this-test e)
         (regexp-match? (pregexp (format "^~a: contract violation" who)) (exn-message e))))

  (test-case "each fit keyword's contract blames the caller"
    (for ([kv (in-list bad-fit-keywords)])
      (check-exn (violates 'fit)
                 (lambda ()
                   (apply-keywords fit (override kv '((#:max-epochs . 1) (#:logger . #f)))
                                   (Good) (ones-loader 2)))
                 (format "~a" (car kv)))))

  (define bad-evaluation-keywords
    (list (list test 'test '((#:logger . #f))
                '((#:device . cpu)
                  (#:logger . stdout)
                  (#:callbacks . (1))
                  (#:limit-test-batches . 2.0)
                  (#:ckpt-path . 1)))
          (list predict 'predict '()
                '((#:device . cpu)
                  (#:callbacks . (1))
                  (#:limit-predict-batches . -1)
                  (#:ckpt-path . 1)))))

  (test-case "each test and predict keyword's contract blames the caller"
    (for ([entry (in-list bad-evaluation-keywords)])
      (match-define (list evaluate who defaults bad) entry)
      (for ([kv (in-list bad)])
        (check-exn (violates who)
                   (lambda ()
                     (apply-keywords evaluate (override kv defaults) (Complete) (ones-loader 2)))
                   (format "~a ~a" who (car kv)))))
    (check-exn (violates 'predict) (lambda () (predict 'trainer (ones-loader 2))))
    (check-exn (violates 'fit) (lambda () (fit 'trainer (ones-loader 2) #:max-epochs 1)))
    (check-exn (violates 'fit) (lambda () (fit (Good) 'loader #:max-epochs 1))))

  (define ((pre-violation message) e)
    (and (blames-this-test e)
         (regexp-match? (regexp-quote message) (exn-message e))))

  (test-case "each #:pre/name of fit, test and predict blames the caller"
    (check-exn (pre-violation "fit needs #:max-epochs or #:max-steps")
               (lambda () (fit (Good) (ones-loader 2) #:logger #f)))
    (for ([row (in-list
                 (list
                  (list "the trainer has no define-training-step" Stepless '())
                  (list "automatic optimization needs define-optimizers" Unoptimized '())
                  (list "a #:val loader needs a define-validation-step" Unvalidated
                        `((#:val . ,(ones-loader 2))))
                  (list "#:limit-train-batches selects no batch from this loader" Good
                        '((#:limit-train-batches . 0)))
                  (list "#:limit-val-batches selects no batch from this loader" Good
                        `((#:val . ,(ones-loader 2)) (#:limit-val-batches . 0.1)))
                  (list interval-message Good '((#:val-check-interval . 3)))
                  (list interval-message Good
                        '((#:val-check-interval . 2) (#:limit-train-batches . 1)))
                  (list (string-append "manual optimization takes no #:accumulate-grad-batches"
                                       " above 1 and no #:gradient-clip-val")
                        Manual '((#:accumulate-grad-batches . 2)))))])
      (match-define (list message make keyword-values) row)
      (check-exn (pre-violation message)
                 (lambda () (apply fit-quiet (make) (ones-loader 2) keyword-values))
                 message))
    (check-exn (pre-violation "the trainer has no define-test-step")
               (lambda () (test (Good) (ones-loader 2) #:logger #f)))
    (check-exn (pre-violation "#:limit-test-batches selects no batch from this loader")
               (lambda () (test (Complete) (ones-loader 2) #:limit-test-batches 0 #:logger #f)))
    (check-exn (pre-violation "the trainer has no define-predict-step")
               (lambda () (predict (Good) (ones-loader 2)))))

  (test-case "what a trainer's own bodies return blames the module defining it"
    (for ([make (list NotOptimizer StrayScheduler TwoOptimizers BadLoss ThreeValues)])
      (check-exn blames-the-trainer (lambda () (fit-quiet (make) (ones-loader 2)))
                 (format "~a" make))))

  (test-case "a batch of the wrong arity names the step"
    (check-exn (lambda (e) (and (exn:fail:contract:arity? e)
                                (regexp-match? #rx"training-step" (exn-message e))))
               (lambda () (fit-quiet (Good) (ones-loader 2 #:columns 3)))))

  (test-case "log! takes a scalar, and only the values its hook allows"
    (check-exn blames-the-trainer (lambda () (fit-quiet (OverLogger) (ones-loader 2))))
    (for ([make (list StepLogger EpochOffLogger)])
      (check-exn (lambda (e) (and (blames-the-trainer e)
                                  (regexp-match? #rx"this hook logs only per epoch"
                                                 (exn-message e))))
                 (lambda () (fit-quiet (make) (ones-loader 2)))
                 (format "~a" make))))

  (test-case "loop control needs a live run in the right hook"
    (define r (fit-quiet (Good) (ones-loader 2)))
    (check-exn blames-this-test (lambda () (request-stop! r)))
    (check-exn blames-this-test (lambda () (skip-step! r)))
    (check-exn blames-this-test (lambda () (skip-epoch! r)))
    (check-exn blames-the-trainer (lambda () (fit-quiet (LoopSkipper) (ones-loader 2)))))

  (test-case "set-hparam! takes a define-hparams name"
    (check-exn blames-this-test (lambda () (set-hparam! (Tuned) 'momentum 0.9))))

  (test-case "two stateful callbacks may not share a state key"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? #rx"two stateful callbacks share the state key \"keeper\""
                                     (exn-message e))))
               (lambda () (fit-quiet (Good) (ones-loader 2)
                                     `(#:callbacks . ,(list (keeper) (keeper))))))))
