#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt"
                    cpu-device cuda-available? cuda-device manual-seed! ones randn requires-grad!
                    tensor->list)
           (only-in "../nn.rkt"
                    Linear learning-rate mse-loss optimizer-param-groups param-group-count
                    param-group-hyperparameters param-group-params parameters
                    sequential-lr set-learning-rate! sgd step-lr)
           (only-in "../train.rkt"
                    define-init define-optimizers define-trainer define-training-step fit
                    run-grad-scaler)
           (only-in "private/blame.rkt" contract-violation? blamed)
           (only-in "private/plain-sgd.rkt" plain-sgd))

  (define eta 0.1)

  (define-trainer Generic (net make-optimizer)
    (define-init (maker) (set! net (Linear 4 3)) (set! make-optimizer maker))
    (define-optimizers (make-optimizer (parameters net)))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define (fit-with maker precision device)
    (manual-seed! 0)
    (define xs (randn 16 4))
    (define ys (randn 16 3))
    (define trainer (Generic maker))
    (define r
      (fit trainer (dataloader (tensor-dataset xs ys) #:batch-size 4)
           #:max-epochs 2 #:logger #f #:device device #:precision precision
           #:accumulate-grad-batches 2 #:gradient-clip-val 0.5))
    (values (map tensor->list (parameters trainer)) (run-grad-scaler r)))

  (define (check-matches-sgd precision device)
    (define-values (ours _scaler) (fit-with (lambda (ps) (plain-sgd ps eta)) precision device))
    (define-values (theirs _sgd-scaler)
      (fit-with (lambda (ps) (sgd ps #:lr eta #:foreach? #f)) precision device))
    (check-equal? ours theirs (format "~a on ~a" precision device)))

  (test-case "G4-L23: fit with a gen:optimizer of one's own steps as sgd does"
    (check-matches-sgd '32-true (cpu-device))
    (check-matches-sgd 'bf16-mixed (cpu-device)))

  (when (cuda-available?)
    (test-case "G4-L23: under '16-mixed the scaler serves a gen:optimizer of one's own"
      (define-values (_ours scaler) (fit-with (lambda (ps) (plain-sgd ps eta)) '16-mixed
                                              (cuda-device)))
      (check-not-false scaler)
      (check-matches-sgd '16-mixed (cuda-device))))

  (define definitions
    '(module half racket/base
       (require torch/nn)
       (struct half ()
         #:methods gen:optimizer
         [(define (optimizer-step! opt) (void))])))

  (test-case "G4-L24 (1): leaving out optimizer-parameters fails at definition"
    (check-exn #rx"optimizer-parameters: required method is not implemented"
               (lambda ()
                 (parameterize ([current-namespace (make-base-namespace)])
                   (eval definitions)
                   (namespace-require ''half)))))

  (define reference-example
    '(module reference racket/base
       (require torch torch/nn)
       (provide stepped groups)
       (struct plain-sgd (params lr)
         #:methods gen:optimizer
         [(define (optimizer-parameters opt) (plain-sgd-params opt))
          (define (optimizer-step! opt)
            (with-no-grad
              (for* ([p (in-list (plain-sgd-params opt))] [g (in-value (maybe-grad p))] #:when g)
                (copy! p (- p (* (plain-sgd-lr opt) g))))))])
       (define w (requires-grad! (ones 2)))
       (define opt (plain-sgd (list w) 0.5))
       (backward! (sum (* w w)))
       (step! opt)
       (define stepped (tensor->list w))
       (define groups (param-group-count opt))))

  (test-case "the reference's plain-sgd steps w by its rate times the gradient"
    (parameterize ([current-namespace (make-base-namespace)])
      (eval reference-example)
      (check-equal? (dynamic-require ''reference 'stepped) '(0.0 0.0))
      (check-equal? (dynamic-require ''reference 'groups) 1)))

  (define w (requires-grad! (ones 2)))
  (define opt (plain-sgd (list w) eta))

  (test-case "G4-L24 (2, 3): one group of the parameters, with no hyperparameters"
    (check-equal? (param-group-count opt) 1)
    (define groups (optimizer-param-groups opt))
    (check-equal? (length groups) 1)
    (check-true (andmap eq? (param-group-params (car groups)) (list w)))
    (check-equal? (param-group-hyperparameters (car groups)) (hasheq)))

  (test-case "G4-L24 (4): the rate readers and the schedules blame the caller"
    (define no-rate?
      (contract-violation? (blamed "optimizer-generic-test.rkt")
                           #rx"this optimizer has no learning rate"))
    (define other (step-lr (sgd (list (requires-grad! (ones 2))) #:lr 0.1) #:step-size 1))
    (for ([call (in-list (list (lambda () (learning-rate opt))
                               (lambda () (set-learning-rate! opt 0.1))
                               (lambda () (step-lr opt #:step-size 1))
                               (lambda () (sequential-lr opt (list other) #:milestones '()))))])
      (check-exn no-rate? call))))
