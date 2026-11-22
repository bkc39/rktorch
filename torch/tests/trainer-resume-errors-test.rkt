#lang racket/base

(module+ test
  (require (only-in racket/port open-output-nowhere)
           (only-in racket/string string-contains?)
           rackunit
           (only-in "../main.rkt" current-generator make-generator)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd step-lr)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" saving-callback Small)
           (only-in "private/trainer-kit.rkt" toy-loader))

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
      (check-exn #rx"shape|size" (lambda () (resume tr path)))
      (check-equal? (trainer-hparams tr) (hasheq 'lr 0.9))))

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

  (define (fit-loop-set field v)
    (lambda (tree)
      (define loop (hash-ref (hash-ref tree "loops") "fit_loop"))
      (hash-set! tree "loops" (hash "fit_loop" (hash-set loop field v)))))

  (define-trainer Tested (net lr)
    (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
    (define-hparams lr)
    (define-test-step (x y) (log! 'test-loss (mse-loss (net x) y))))

  (test-case "T3-L3: a malformed tree is refused before any hook, naming the path and the key"
    (with-temporary-directory (dir #:template "malformed-~a")
      (for ([edit (list (lambda (tree) (hash-remove! tree "loops"))
                        (lambda (tree) (hash-set! tree "loops" (hash)))
                        (fit-loop-set "phase" 3)
                        (fit-loop-set "eval_batches" "many")
                        (lambda (tree) (hash-remove! tree "trainer_hparams"))
                        (lambda (tree) (hash-set! tree "trainer_hparams" '(lr)))
                        (lambda (tree) (hash-remove! tree "global_step")))]
            [key (list "loops" "loops" "phase" "eval_batches" "trainer_hparams" "trainer_hparams"
                       "global_step")]
            [i (in-naturals)])
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
      (define path (saved-with dir (list (reshaping "test" (lambda (tree)
                                                             (hash-remove! tree "trainer_hparams"))))
                               #:name "test.ckpt"))
      (check-names-path #rx"cannot be resumed.*trainer_hparams" path
                        (lambda () (test (Tested) (toy-loader 2) #:logger #f #:ckpt-path path))
                        "test #:ckpt-path"))))
