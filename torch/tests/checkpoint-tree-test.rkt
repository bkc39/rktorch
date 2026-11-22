#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" current-generator make-generator rng-state)
           (only-in "../nn.rkt"
                    Linear load-optimizer-state! mse-loss optimizer-state-dict parameters
                    scheduler-state-dict sgd step-lr)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" call-with-live-run Small)
           (only-in "private/comparable.rkt" comparable host-bytes)
           (only-in "private/trainer-kit.rkt" toy-loader))

  (define-trainer Scheduled (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
      (values opt (step-lr opt #:step-size 1 #:gamma 0.5)))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y)))

  (define-trainer Spread (net)
    (define-init (width #:bias [b? #t] . more) (set! net (Linear 2 width #:bias? b?)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y)))

  (define (saved-at path #:trainer [tr (Small)] #:callbacks [callbacks '()]
                    #:weights-only? [weights-only? #f])
    (define seen (box #f))
    (void (call-with-live-run
           #:trainer tr #:callbacks callbacks
           (lambda (r)
             (save-checkpoint! r path #:weights-only? weights-only?)
             (set-box! seen r))))
    (values (read-checkpoint path) (unbox seen)))

  (test-case "LO-9: the training state is saved as the nn accessors give it"
    (with-temporary-directory (dir #:template "tree-~a")
      (define tr (Scheduled))
      (define-values (tree r) (saved-at (build-path dir "s.ckpt") #:trainer tr))
      (define opt (car (run-optimizers r)))
      (define sched (car (run-schedulers r)))
      (check-equal? (comparable (hash-ref tree "optimizer_states"))
                    (comparable (list (optimizer-state-dict opt tr))))
      (check-equal? (hash-ref tree "lr_schedulers")
                    (list (scheduler-state-dict (scheduler-config-scheduler sched))))
      (check-false (hash-has-key? tree "MixedPrecision") "no grad scaler, no entry")
      (check-equal? (hash-ref tree "trainer_hparams") (hash))
      (define loop (hash-ref (hash-ref tree "loops") "fit_loop"))
      (check-equal? (sort (hash-keys loop) string<?)
                    '("batches_that_stepped" "epoch" "eval_batches" "phase"))
      (check-equal? (map (lambda (k) (hash-ref loop k)) '("epoch" "phase" "batches_that_stepped"))
                    '(1 "fit-end" 2))
      (define fresh (sgd (parameters tr) #:lr 0.5))
      (load-optimizer-state! fresh tr (car (hash-ref tree "optimizer_states")))
      (check-equal? (comparable (optimizer-state-dict fresh tr))
                    (comparable (optimizer-state-dict opt tr))
                    "the saved optimizer state loads back")))

  (test-case "LO-9: a mid-epoch save records its phase and batch"
    (with-temporary-directory (dir #:template "phase-~a")
      (define path (build-path dir "b.ckpt"))
      (define saver
        (lambda-callback
         #:on-train-batch-end
         (lambda (r . _)
           (when (= (run-global-step r) 1) (save-checkpoint! r path)))))
      (void (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f #:callbacks (list saver)))
      (define loop (hash-ref (hash-ref (read-checkpoint path) "loops") "fit_loop"))
      (check-equal? (hash-ref loop "phase") "batch-end")
      (check-equal? (hash-ref loop "batch_index") 0)
      (check-equal? (hash-ref loop "epoch") 0)))

  (test-case "weights-only keeps the model, hyperparameters and position, as Lightning's"
    (with-temporary-directory (dir #:template "weights-~a")
      (define-values (tree _r) (saved-at (build-path dir "w.ckpt") #:weights-only? #t))
      (check-equal? (sort (hash-keys tree) string<?)
                    '("callback_metrics" "epoch" "global_step" "hyper_parameters"
                      "init_dropped" "init_formals" "loops" "state_dict" "trainer_hparams"))))

  (test-case "the RNG bundle holds the CPU stream, current-generator and Racket's PRNG"
    (with-temporary-directory (dir #:template "rng-~a")
      (define path (build-path dir "r.ckpt"))
      (define expected (box #f))
      (define (capture-then-save r)
        (set-box! expected (list (host-bytes (rng-state))
                                 (vector->list (pseudo-random-generator->vector
                                                (current-pseudo-random-generator)))))
        (save-checkpoint! r path))
      (void (call-with-live-run capture-then-save))
      (define rng (hash-ref (read-checkpoint path) "rng"))
      (check-equal? (list (host-bytes (hash-ref rng "cpu")) (hash-ref rng "racket"))
                    (unbox expected))
      (check-false (hash-ref rng "current_generator"))
      (parameterize ([current-generator (make-generator 7)])
        (void (call-with-live-run (lambda (r) (save-checkpoint! r path)))))
      (check-equal? (hash-ref (hash-ref (hash-ref (read-checkpoint path) "rng") "current_generator")
                              "device")
                    'cpu)))

  (test-case "B6: the init formals keep their shape; the file goes where the path says"
    (with-temporary-directory (dir #:template "formals-~a")
      (define path (build-path dir "deeper" "still" "f.ckpt"))
      (define-values (tree _r) (saved-at path #:trainer (Spread 3 #:bias #f 'x "y")))
      (check-equal? (hash-ref tree "init_formals")
                    '(("positional" "width") ("keyword" "#:bias" "b?") ("rest" "more")))
      (check-equal? (hash-ref tree "hyper_parameters") (hash "width" 3 "b?" #f "more" '(x "y")))
      (check-equal? (hash-ref tree "init_dropped") '())
      (define seen (box #f))
      (void (load-from-checkpoint (lambda (w #:bias b . more) (set-box! seen (list* w b more))
                                    (Spread w #:bias b))
                                  path
                                  #:init-args (hasheq 'width 3 'unknown 1)))
      (check-equal? (unbox seen) '(3 #f x "y")
                    "an #:init-args name that is no formal is ignored, as Lightning's")))

  (test-case "a callback state-dict that is not a checkpoint value names the callback"
    (with-temporary-directory (dir #:template "bad-state-~a")
      (define bad (lambda-callback #:state-key "holder" #:state-dict (lambda (_r) (vector 1))))
      (check-exn (lambda (e) (and (exn:fail:contract? e)
                                  (regexp-match? #rx"state-dict" (exn-message e))
                                  (regexp-match? #rx"\"holder\"" (exn-message e))))
                 (lambda () (call-with-live-run #:callbacks (list bad)
                                                (lambda (r) (save-checkpoint! r (build-path dir "x.ckpt"))))))
      (check-equal? (directory-list dir) '()))))
