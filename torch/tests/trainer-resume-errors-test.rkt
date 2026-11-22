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
      (check-equal? (trainer-hparams tr) (hasheq 'lr 0.9)))))
