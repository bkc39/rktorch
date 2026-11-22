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
                        "no current-generator"))))
