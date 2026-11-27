#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           (only-in racket/port open-output-nowhere)
           (only-in racket/string string-contains?)
           rackunit
           (only-in "../main.rkt" current-generator make-generator)
           (only-in "../nn.rkt" Linear adamw mse-loss parameters sgd step-lr)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" saving-callback Small)
           (only-in "private/trainer-kit.rkt" call-with-logged-warnings toy-loader))

  (define (saved-with dir callbacks #:name [name "e.ckpt"] #:max-epochs [epochs 1]
                      #:weights-only? [weights-only? #f])
    (define path (build-path dir name))
    (void (fit (Small) (toy-loader 4) #:max-epochs epochs #:logger #f
               #:callbacks (append callbacks
                                   (list (saving-callback (lambda (_r) path)
                                                          #:weights-only? weights-only?)))))
    path)

  (define (resume tr path #:max-epochs [epochs 2] #:callbacks [callbacks '()])
    (parameterize ([current-error-port (open-output-nowhere)])
      (fit tr (toy-loader 4) #:max-epochs epochs #:logger #f #:ckpt-path path
           #:callbacks callbacks)))

  (define (check-names-path rx path thunk [message ""])
    (check-exn (lambda (e) (and (regexp-match? rx (exn-message e))
                                (string-contains? (exn-message e) (path->string path))))
               thunk
               message))

  (define-trainer Tuned (net lr)
    (define-init () (set! net (Linear 2 3)) (set! lr 0.9))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "T3-L18: a model that does not load leaves the hyperparameters as they were"
    (with-temporary-directory (dir #:template "hparams-~a")
      (define path (saved-with dir '()))
      (define tr (Tuned))
      (check-names-path #rx"shape|size" path (lambda () (resume tr path)))
      (check-equal? (trainer-hparams tr) (hasheq 'lr 0.9))))

  (define-trainer Adaptive (net lr)
    (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
    (define-hparams lr)
    (define-optimizers (adamw (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Pair (net)
    #:automatic-optimization? #f
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (values (sgd (parameters net) #:lr 0.1) (sgd (parameters net) #:lr 0.1)))
    (define-training-step (x y)
      (for ([opt (in-list (run-optimizers run))])
        (manual-zero-grads! run opt)
        (manual-backward! run (mse-loss (net x) y))
        (manual-step! run opt))
      #f))

  (define-trainer Scheduled (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1))
      (values opt (step-lr opt #:step-size 1)))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "every resume error names the checkpoint's path"
    (with-temporary-directory (dir #:template "named-~a")
      (define path (saved-with dir '()))
      (check-names-path #rx"optimizer.*1.*2" path (lambda () (resume (Pair) path)) "optimizers")
      (check-names-path #rx"does not fit the optimizer" path (lambda () (resume (Adaptive) path))
                        "another optimizer kind")
      (check-names-path #rx"scheduler.*0.*1" path (lambda () (resume (Scheduled) path))
                        "schedulers")
      (define weights (saved-with dir '() #:name "weights.ckpt" #:weights-only? #t))
      (check-names-path #rx"contains only the model" weights (lambda () (resume (Small) weights))
                        "weights-only")
      (define later (saved-with dir '() #:name "later.ckpt" #:max-epochs 2))
      (check-names-path #rx"past #:max-epochs 1" later
                        (lambda () (resume (Small) later #:max-epochs 1))
                        "past #:max-epochs")
      (define generated
        (parameterize ([current-generator (make-generator 13)])
          (saved-with dir '() #:name "generated.ckpt")))
      (check-names-path #rx"current-generator" generated (lambda () (resume (Small) generated))
                        "no current-generator")))

  (define (reshaping key edit)
    (lambda-callback #:state-key (format "reshape-~a" key)
                     #:on-save-checkpoint (lambda (_r tree) (edit tree))))

  (define ((fit-loop-set field v) tree)
    (define loop (hash-ref (hash-ref tree "loops") "fit_loop"))
    (hash-set! tree "loops" (hash "fit_loop" (hash-set loop field v))))

  (define-trainer Tested (net lr)
    (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
    (define-hparams lr)
    (define-test-step (x y) (log! 'test-loss (mse-loss (net x) y))))

  (test-case "T3-L3: a malformed tree is refused before any hook, naming the path and the key"
    (with-temporary-directory (dir #:template "malformed-~a")
      (define ((set-entry key v) tree) (hash-set! tree key v))
      (define ((remove-entry key) tree) (hash-remove! tree key))
      (for ([malformed (in-list
                        (list (cons "loops" (remove-entry "loops"))
                              (cons "loops" (set-entry "loops" (hash)))
                              (cons "phase" (fit-loop-set "phase" 3))
                              (cons "eval_batches" (fit-loop-set "eval_batches" "many"))
                              (cons "epoch_end_logged" (fit-loop-set "epoch_end_logged" 1))
                              (cons "trainer_hparams" (remove-entry "trainer_hparams"))
                              (cons "trainer_hparams" (set-entry "trainer_hparams" '(lr)))
                              (cons "global_step" (remove-entry "global_step"))
                              (cons "callback_metrics"
                                    (set-entry "callback_metrics" (hash "x" "y")))
                              (cons "rng" (remove-entry "rng"))
                              (cons "metric_book" (set-entry "metric_book" (hash "train" 1)))
                              (cons "optimizer_states" (set-entry "optimizer_states" (hash)))
                              (cons "lr_schedulers" (set-entry "lr_schedulers" 0))))]
            [i (in-naturals)])
        (match-define (cons key edit) malformed)
        (define path (saved-with dir (list (reshaping i edit)) #:name (format "~a.ckpt" i)))
        (define seen (box '()))
        (check-names-path (regexp (string-append "cannot be resumed.*key: \"[a-z_.]*" key))
                          path
                          (lambda ()
                            (resume (Small) path
                                    #:callbacks (list (lambda-callback
                                                       #:setup (lambda (_r _stage)
                                                                 (set-box! seen '(setup)))))))
                          key)
        (check-equal? (unbox seen) '() (format "~a: no hook ran" key)))
      (define path (saved-with dir (list (reshaping "test" (remove-entry "trainer_hparams")))
                               #:name "test.ckpt"))
      (check-names-path #rx"cannot be resumed.*trainer_hparams" path
                        (lambda () (test (Tested) (toy-loader 2) #:logger #f #:ckpt-path path))
                        "test #:ckpt-path")))

  (test-case "an epoch-end file without epoch_end_logged resumes without a warning"
    (with-temporary-directory (dir #:template "older-~a")
      (define (drop-logged tree)
        (define loop (hash-ref (hash-ref tree "loops") "fit_loop"))
        (hash-set! tree "loops" (hash "fit_loop" (hash-remove loop "epoch_end_logged"))))
      (define path (saved-with dir (list (reshaping "older" drop-logged))))
      (define-values (_run warnings) (call-with-logged-warnings (lambda () (resume (Small) path))))
      (check-equal? warnings '()))))
