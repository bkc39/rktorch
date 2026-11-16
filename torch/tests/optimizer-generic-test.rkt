#lang racket/base

(module+ test
  (require (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt"
                    cpu-device cuda-available? cuda-device manual-seed! ones randn requires-grad!
                    tensor->list)
           (only-in "../nn.rkt"
                    Linear cosine-annealing-lr cosine-annealing-warm-restarts exponential-lr
                    lambda-lr learning-rate linear-lr mse-loss multi-step-lr one-cycle-lr
                    optimizer-param-groups param-group-count param-group-hyperparameters
                    param-group-params parameters sequential-lr set-learning-rate! sgd step-lr)
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
              (for ([p (in-list (plain-sgd-params opt))] #:when (has-grad? p))
                (copy! p (- p (* (plain-sgd-lr opt) (grad p)))))))])
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

  (define no-rate?
    (contract-violation? (blamed "optimizer-generic-test.rkt")
                         #rx"expected: \"optimizer with a learning rate\""))

  (define (rate-calls opt)
    (define other (step-lr (sgd (list (requires-grad! (ones 2))) #:lr 0.1) #:step-size 1))
    `((learning-rate . ,(lambda () (learning-rate opt)))
      (set-learning-rate! . ,(lambda () (set-learning-rate! opt 0.1)))
      (step-lr . ,(lambda () (step-lr opt #:step-size 1)))
      (multi-step-lr . ,(lambda () (multi-step-lr opt #:milestones '(1))))
      (exponential-lr . ,(lambda () (exponential-lr opt #:gamma 0.9)))
      (cosine-annealing-lr . ,(lambda () (cosine-annealing-lr opt #:t-max 2)))
      (linear-lr . ,(lambda () (linear-lr opt)))
      (one-cycle-lr . ,(lambda () (one-cycle-lr opt #:max-lr 0.1 #:total-steps 4)))
      (lambda-lr . ,(lambda () (lambda-lr opt (lambda (_t) 1.0))))
      (sequential-lr . ,(lambda () (sequential-lr opt (list other) #:milestones '())))
      (cosine-annealing-warm-restarts
       . ,(lambda () (cosine-annealing-warm-restarts opt #:t-0 2)))))

  (test-case "G4-L24 (4): the rate readers and every schedule blame the caller"
    (for ([call (in-list (rate-calls opt))])
      (check-exn no-rate? (cdr call) (symbol->string (car call)))))

  (define-runtime-path main-path "../main.rkt")
  (define-runtime-path nn-path "../nn.rkt")

  (define rooted
    `(module rooted racket/base
       (require (file ,(path->string main-path)) (file ,(path->string nn-path)))
       (provide opt)
       (define inner (sgd (list (requires-grad! (ones 2))) #:lr 0.1))
       (struct rooted (params)
         #:methods gen:optimizer
         [(define (optimizer-parameters opt) (rooted-params opt))
          (define (optimizer-step! opt) (void))
          (define (optimizer-root opt) inner)
          (define (optimizer-store opt) inner)])
       (define opt (rooted (list (requires-grad! (ones 2)))))))

  (test-case "a gen:optimizer of one's own cannot borrow another optimizer's rates"
    (define here (variable-reference->namespace (#%variable-reference)))
    (parameterize ([current-namespace (make-base-namespace)])
      (namespace-attach-module here main-path)
      (namespace-attach-module here nn-path)
      (eval rooted)
      (define opt (dynamic-require ''rooted 'opt))
      (check-equal? (param-group-count opt) 1)
      (for ([call (in-list (rate-calls opt))])
        (check-exn no-rate? (cdr call) (symbol->string (car call)))))))
