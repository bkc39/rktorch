#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" manual-seed! randn tensor->list)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/comparable.rkt" comparable)
           (only-in "private/trainer-kit.rkt" collecting-logger collecting-steps toy-loader))

  (define-trainer Tested (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-test-step (x y) (log! 'test-loss (mse-loss (net x) y)))
    (define-predict-step (x _y) (net x)))

  (define (test-rows tr #:ckpt-path [path #f])
    (define l (collecting-logger))
    (void (test tr (toy-loader 4) #:logger l #:ckpt-path path))
    (comparable (collecting-steps l)))

  (test-case "T3-L23: test #:ckpt-path gives the trained rows and leaves the stream alone"
    (with-temporary-directory (dir #:template "test-ckpt-~a")
      (define path (build-path dir "trained.ckpt"))
      (define trained (Tested))
      (void (fit trained (toy-loader 4) #:max-epochs 3 #:logger #f
                 #:callbacks (list (lambda-callback
                                    #:on-train-end (lambda (r) (save-checkpoint! r path))))))
      (define-values (fresh resumed plain) (values (Tested) (Tested) (Tested)))
      (check-equal? (test-rows fresh #:ckpt-path path) (test-rows trained))
      (manual-seed! 7)
      (void (test-rows resumed #:ckpt-path path))
      (define after-resume (tensor->list (randn 3)))
      (manual-seed! 7)
      (void (test-rows plain))
      (check-equal? after-resume (tensor->list (randn 3)) "no RNG restore")))

  (test-case "LO-11: predict #:ckpt-path gives the trained trainer's outputs"
    (with-temporary-directory (dir #:template "predict-ckpt-~a")
      (define path (build-path dir "trained.ckpt"))
      (define trained (Tested))
      (void (fit trained (toy-loader 4) #:max-epochs 3 #:logger #f
                 #:callbacks (list (lambda-callback
                                    #:on-train-end (lambda (r) (save-checkpoint! r path))))))
      (define (outputs tr #:ckpt-path [path #f])
        (map tensor->list (predict tr (toy-loader 4) #:ckpt-path path)))
      (check-equal? (outputs (Tested) #:ckpt-path path) (outputs trained))
      (check-not-equal? (outputs (Tested)) (outputs trained) "the restore is what made them equal"))))
