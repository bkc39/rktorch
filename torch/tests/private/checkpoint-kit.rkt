#lang racket/base

(require (only-in "../../data/loader.rkt" dataloader tensor-dataset)
         (only-in "../../main.rkt" generator-state make-generator manual-seed! randn rng-state)
         (only-in "../../nn.rkt" Dropout Linear Sequential mse-loss parameters sgd)
         "../../train.rkt"
         (only-in "comparable.rkt" host-bytes layer-bytes)
         (only-in "trainer-kit.rkt" toy-loader))

(provide call-with-live-run
         fit-draws
         Small
         saving-callback
         stored)

(define-trainer Small (net lr)
  (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
  (define-hparams lr)
  (define-optimizers (sgd (parameters net) #:lr lr))
  (define-training-step (x y) (mse-loss (net x) y))
  (define-validation-step (x y) (log! 'val-loss (mse-loss (net x) y))))

(define-trainer Drawing (net)
  (define-init () (set! net (Sequential (Linear 2 4) (Dropout #:p 0.5) (Linear 4 1))))
  (define-optimizers (sgd (parameters net) #:lr 0.1))
  (define-training-step (x y) (mse-loss (net x) y))
  (define-validation-step (x y) (log! 'val-loss (mse-loss (net x) y))))

(define (fit-draws callbacks #:epochs [epochs 3])
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (manual-seed! 0)
    (random-seed 0)
    (define seeded (host-bytes (rng-state)))
    (define source (make-generator 3))
    (define shuffle (make-generator 5))
    (define (data n) (tensor-dataset (randn n 2 #:generator source) (randn n 1 #:generator source)))
    (define tr (Drawing))
    (void (fit tr (dataloader (data 8) #:batch-size 2 #:shuffle? #t #:generator shuffle)
               #:val (dataloader (data 4) #:batch-size 2 #:shuffle? #t)
               #:max-epochs epochs #:logger #f #:callbacks callbacks))
    (hasheq 'parameters (layer-bytes tr)
            'seeded-rng seeded
            'cpu-rng (host-bytes (rng-state))
            'racket-rng (pseudo-random-generator->vector (current-pseudo-random-generator))
            'loader-generator (host-bytes (generator-state shuffle)))))

(define (call-with-live-run proc #:trainer [trainer (Small)] #:callbacks [callbacks '()]
                            #:max-epochs [epochs 1])
  (fit trainer (toy-loader 4) #:val (toy-loader 2) #:max-epochs epochs #:logger #f
       #:callbacks (append callbacks (list (lambda-callback #:on-train-end proc)))))

(define (stored value-box)
  (lambda-callback #:state-key "stored"
                   #:on-save-checkpoint
                   (lambda (_r ckpt) (hash-set! ckpt "probe" (unbox value-box)))))

(define (saving-callback path-of #:weights-only? [weights-only? #f])
  (lambda-callback #:state-key "saving"
                   #:on-train-epoch-end
                   (lambda (r) (save-checkpoint! r (path-of r) #:weights-only? weights-only?))))
