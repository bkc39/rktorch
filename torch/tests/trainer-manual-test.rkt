#lang racket/base

(module+ test
  (require (only-in racket/list take)
           (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../data/loader.rkt" dataloader in-dataloader tensor-dataset)
           (only-in "../main.rkt"
                    - autocast-cache-enabled? backward! manual-seed! randn reshape tensor->list
                    with-autocast)
           (only-in "../nn.rkt"
                    Linear layer-copy learning-rate mse-loss parameters sgd step! step-lr
                    zero-grads!)
           "../train.rkt"
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/lightning-names.rkt" expected-logged-steps)
           (only-in "private/trainer-kit.rkt"
                    collecting-logger collecting-rows step-each! toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "trainer-manual-test"))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (loader)
    (manual-seed! 1)
    (dataloader (tensor-dataset (randn 8 2) (randn 8 1)) #:batch-size 2))

  (define (plain-forward thunk)
    (thunk))

  (define (bf16-forward thunk)
    (with-autocast #:device 'cpu #:dtype 'bfloat16 (thunk)))

  (define-trainer Gan (gen disc opts wrap)
    #:automatic-optimization? #f
    (define-init (g d sink [w plain-forward])
      (set! gen g)
      (set! disc d)
      (set! opts sink)
      (set! wrap w))
    (define-optimizers
      (define g-opt (sgd (parameters gen) #:lr 0.1))
      (define d-opt (sgd (parameters disc) #:lr 0.05))
      (set-box! opts (list g-opt d-opt))
      (values g-opt d-opt (step-lr g-opt #:step-size 1 #:gamma 0.5)))
    (define-training-step (x y)
      (match-define (list g-opt d-opt) (unbox opts))
      (manual-zero-grads! run d-opt)
      (manual-backward! run (wrap (lambda () (~> (gen x) disc (mse-loss y)))))
      (manual-step! run d-opt)
      (manual-zero-grads! run g-opt)
      (define g-loss (wrap (lambda () (mse-loss (disc (gen x)) (- y)))))
      (manual-backward! run g-loss)
      (manual-step! run g-opt)
      g-loss))

  (define (hand-gan gen disc #:forward [forward plain-forward])
    (define g-opt (sgd (parameters gen) #:lr 0.1))
    (define d-opt (sgd (parameters disc) #:lr 0.05))
    (for* ([_ (in-range 2)] [(x y) (in-dataloader (loader))])
      (zero-grads! d-opt)
      (backward! (forward (lambda () (~> (gen x) disc (mse-loss y)))))
      (step! d-opt)
      (zero-grads! g-opt)
      (backward! (forward (lambda () (mse-loss (disc (gen x)) (- y)))))
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

  (bounded-test "a bf16-mixed manual GAN casts each forward afresh, as Lightning"
    (manual-seed! 0)
    (define gen (Linear 2 2))
    (define disc (Linear 2 1))
    (define gen* (layer-copy gen))
    (define disc* (layer-copy disc))
    (fit (Gan gen disc (box #f)) (loader) #:max-epochs 2 #:logger #f #:precision 'bf16-mixed)
    (check-equal? (map flat (append (parameters gen) (parameters disc)))
                  (hand-gan gen* disc* #:forward bf16-forward))
    (check-true (autocast-cache-enabled?)))

  (bounded-test "a with-autocast inside a bf16-mixed step keeps fit's cache off"
    (manual-seed! 0)
    (define gen (Linear 2 2))
    (define disc (Linear 2 1))
    (define gen* (layer-copy gen))
    (define disc* (layer-copy disc))
    (fit (Gan gen disc (box #f) bf16-forward) (loader)
         #:max-epochs 2 #:logger #f #:precision 'bf16-mixed)
    (check-equal? (map flat (append (parameters gen) (parameters disc)))
                  (hand-gan gen* disc* #:forward bf16-forward)))

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
               (lambda () (fit (Skipper) (toy-loader 2) #:max-epochs 1 #:logger #f))))

  (define-trainer Pairs (net)
    #:automatic-optimization? #f
    (define-init (n) (set! net n))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (manual-backward! run (mse-loss (net x) y))
      (when (odd? (run-batch-index run))
        (match-define (list opt) (run-optimizers run))
        (manual-step! run opt)
        (manual-zero-grads! run opt))
      #f)
    (define-validation-step (x y)
      (mse-loss (net x) y)))

  (define (hand-pairs net #:clear-each-batch? clear?)
    (define opt (sgd (parameters net) #:lr 0.1))
    (for ([(x y) (in-dataloader (loader))]
          [i (in-naturals)])
      (~> (net x) (mse-loss y) backward!)
      (when (odd? i)
        (step! opt))
      (when (or clear? (odd? i))
        (zero-grads! opt)))
    (map flat (parameters net)))

  (bounded-test "manual mode clears the gradients before an in-epoch validation, as Lightning"
    (manual-seed! 0)
    (define net (Linear 2 1))
    (define validated (layer-copy net))
    (match-define (list cleared kept)
      (map (lambda (clear?) (hand-pairs (layer-copy net) #:clear-each-batch? clear?)) '(#t #f)))
    (fit (Pairs net) (loader) #:max-epochs 1 #:logger #f)
    (fit (Pairs validated) (loader) #:val (toy-loader 2) #:val-check-interval 1
         #:max-epochs 1 #:logger #f)
    (check-equal? (map flat (parameters net)) kept)
    (check-equal? (map flat (parameters validated)) cleared)
    (check-not-equal? cleared kept))

  (define-trainer ManualLogged (net steps)
    #:automatic-optimization? #f
    (define-init (n) (set! net (Linear 2 1)) (set! steps n))
    (define-optimizers (values (sgd (parameters net) #:lr 0.1) (sgd (parameters net) #:lr 0.1)))
    (define-training-step (x y)
      (log! 'tl 1.0 #:on-step? #t #:on-epoch? #t)
      (step-each! run (take (run-optimizers run) steps) (lambda () (mse-loss (net x) y)))
      #f)
    (define-validation-step (x y)
      (log! 'vl 2.0)
      (mse-loss (net x) y)))

  (define (check-manual-rows name steps #:log-every-n-steps [log-every 1])
    (define l (collecting-logger))
    (fit (ManualLogged steps) (toy-loader 6) #:val (toy-loader 4) #:max-epochs 2
         #:num-sanity-val-steps 1 #:log-every-n-steps log-every #:logger l)
    (check-equal? (collecting-rows l) (expected-logged-steps name) (format "~a" name)))

  (bounded-test "manual rows are labelled and paced by batches, as Lightning's"
    (check-manual-rows 'manual 2)
    (check-manual-rows 'manual-log-every-4 2 #:log-every-n-steps 4)
    (check-manual-rows 'manual-no-steps 0)))
