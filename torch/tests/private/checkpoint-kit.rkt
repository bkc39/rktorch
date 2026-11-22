#lang racket/base

(require (only-in "../../nn.rkt" Linear mse-loss parameters sgd)
         "../../train.rkt"
         (only-in "comparable.rkt" comparable)
         (only-in "trainer-kit.rkt" toy-loader))

(provide call-with-live-run
         decoded-form
         Small
         saving-callback
         stored)

(define-trainer Small (net lr)
  (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
  (define-hparams lr)
  (define-optimizers (sgd (parameters net) #:lr lr))
  (define-training-step (x y) (mse-loss (net x) y))
  (define-validation-step (x y) (log! 'val-loss (mse-loss (net x) y))))

(define (call-with-live-run proc #:trainer [trainer (Small)] #:callbacks [callbacks '()]
                            #:max-epochs [epochs 1])
  (fit trainer (toy-loader 4) #:val (toy-loader 2) #:max-epochs epochs #:logger #f
       #:callbacks (append callbacks (list (lambda-callback #:on-train-end proc)))))

(define (stored value-box)
  (lambda-callback #:state-key "stored"
                   #:on-save-checkpoint (lambda (_r ckpt) (hash-set! ckpt "probe" (unbox value-box)))))

(define (saving-callback path-of #:weights-only? [weights-only? #f])
  (lambda-callback #:state-key "saving"
                   #:on-train-epoch-end
                   (lambda (r) (save-checkpoint! r (path-of r) #:weights-only? weights-only?))))

(define (decoded-form v)
  (comparable v #:hash-kinds? #t))
