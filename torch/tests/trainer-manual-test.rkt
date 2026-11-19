#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../data/loader.rkt" dataloader in-dataloader tensor-dataset)
           (only-in "../main.rkt" - backward! manual-seed! randn reshape tensor->list)
           (only-in "../nn.rkt"
                    Linear layer-copy learning-rate mse-loss parameters sgd step! step-lr
                    zero-grads!)
           "../train.rkt"
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/trainer-kit.rkt" toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "trainer-manual-test"))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (loader)
    (manual-seed! 1)
    (dataloader (tensor-dataset (randn 8 2) (randn 8 1)) #:batch-size 2))

  (define-trainer Gan (gen disc opts)
    #:automatic-optimization? #f
    (define-init (g d sink) (set! gen g) (set! disc d) (set! opts sink))
    (define-optimizers
      (define g-opt (sgd (parameters gen) #:lr 0.1))
      (define d-opt (sgd (parameters disc) #:lr 0.05))
      (set-box! opts (list g-opt d-opt))
      (values g-opt d-opt (step-lr g-opt #:step-size 1 #:gamma 0.5)))
    (define-training-step (x y)
      (match-define (list g-opt d-opt) (unbox opts))
      (manual-zero-grads! run d-opt)
      (manual-backward! run (~> (gen x) disc (mse-loss y)))
      (manual-step! run d-opt)
      (manual-zero-grads! run g-opt)
      (define g-loss (mse-loss (disc (gen x)) (- y)))
      (manual-backward! run g-loss)
      (manual-step! run g-opt)
      g-loss))

  (define (hand-gan gen disc)
    (define g-opt (sgd (parameters gen) #:lr 0.1))
    (define d-opt (sgd (parameters disc) #:lr 0.05))
    (for* ([_ (in-range 2)] [(x y) (in-dataloader (loader))])
      (zero-grads! d-opt)
      (~> (gen x) disc (mse-loss y) backward!)
      (step! d-opt)
      (zero-grads! g-opt)
      (backward! (mse-loss (disc (gen x)) (- y)))
      (step! g-opt))
    (map flat (append (parameters gen) (parameters disc))))

  (bounded-test "G3-L20: a manual GAN steps as a hand loop, and the loop steps no schedule"
    (manual-seed! 0)
    (define gen (Linear 2 2))
    (define disc (Linear 2 1))
    (define gen* (layer-copy gen))
    (define disc* (layer-copy disc))
    (define opts (box #f))
    (define r (fit (Gan gen disc opts) (loader) #:max-epochs 2 #:logger #f))
    (check-equal? (map flat (append (parameters gen) (parameters disc))) (hand-gan gen* disc*))
    (check-equal? (learning-rate (car (unbox opts))) 0.1)
    (check-equal? (run-global-step r) 16 "G3-L19: two optimizer steps per batch"))

  (define-trainer Stray (net stray)
    #:automatic-optimization? #f
    (define-init ()
      (set! net (Linear 2 1))
      (set! stray (~> (Linear 2 1) parameters (sgd #:lr 0.1))))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (manual-step! run stray)
      #f))

  (define-trainer Skipper (net)
    #:automatic-optimization? #f
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (skip-step! run)
      #f))

  (bounded-test "G3-L21: an optimizer define-optimizers did not return blames the caller"
    (check-exn blames-this-test?
               (lambda () (fit (Stray) (toy-loader 2) #:max-epochs 1 #:logger #f))))

  (bounded-test "G3-L21: manual-step! outside a manual training step blames the caller"
    (define opts (box #f))
    (define r (fit (Gan (Linear 2 2) (Linear 2 1) opts) (loader) #:max-epochs 1 #:logger #f))
    (check-exn blames-this-test? (lambda () (manual-step! r (car (unbox opts))))))

  (bounded-test "G3-L21: skip-step! in manual mode blames the caller"
    (check-exn blames-this-test?
               (lambda () (fit (Skipper) (toy-loader 2) #:max-epochs 1 #:logger #f)))))
