#lang racket/base

(module+ test
  (require rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * + backward! cuda-available? grad manual-seed! randn reshape sum tensor
                    tensor->list)
           (only-in "../nn.rkt"
                    Linear Parameter adam grad-scaler grad-scaler-enabled? grad-scaler-scale
                    grad-scaler-scale-loss grad-scaler-step! grad-scaler-unscale!
                    grad-scaler-update! parameters sgd step!)
           (only-in "private/within.rkt" bounded-test))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (cpu-scaler)
    (grad-scaler #:device 'cpu))

  (unless (cuda-available?)
    (bounded-test "a 'cuda scaler without CUDA is disabled"
      (check-false (grad-scaler-enabled? (grad-scaler)))))

  (bounded-test "the scale before any loss is scaled is the initial scale"
    (check-equal? (grad-scaler-scale (grad-scaler #:device 'cpu #:init-scale 8)) 8.0))

  (bounded-test "an update to a given scale sets it"
    (define p (Parameter (tensor '(0.5 -0.25))))
    (define opt (sgd (list p) #:lr 0.1))
    (define s (cpu-scaler))
    (backward! (grad-scaler-scale-loss s (sum (* p p))))
    (grad-scaler-step! s opt)
    (grad-scaler-update! s 4.0)
    (check-equal? (grad-scaler-scale s) 4.0))

  (define (fused-run scaled? bad?)
    (manual-seed! 0)
    (define net (Linear 3 2))
    (define x (randn 4 3))
    (define opt (adam (parameters net) #:fused? #t #:lr 0.1))
    (define loss (* (sum (net x)) (if bad? +inf.0 1.0)))
    (cond
      [scaled?
       (define s (cpu-scaler))
       (backward! (grad-scaler-scale-loss s loss))
       (grad-scaler-unscale! s opt)
       (grad-scaler-step! s opt)
       (grad-scaler-update! s)]
      [else
       (backward! loss)
       (step! opt)])
    (map flat (parameters net)))

  (bounded-test "a fused step after an explicit unscale equals the plain fused step"
    (check-equal? (fused-run #t #f) (fused-run #f #f)))

  (bounded-test "a fused step after an explicit unscale skips an inf step"
    (manual-seed! 0)
    (define before (map flat (parameters (Linear 3 2))))
    (check-equal? (fused-run #t #t) before))

  (bounded-test "two dtypes on one device share one inf check"
    (define p32 (Parameter (tensor '(0.5 -0.25))))
    (define p64 (Parameter (tensor '(1.5 2.0) #:dtype 'float64)))
    (define opt (sgd (list p32 p64) #:lr 0.1))
    (define s (cpu-scaler))
    (backward! (grad-scaler-scale-loss s (+ (sum (* p32 p32)) (sum (* p64 p64)))))
    (grad-scaler-unscale! s opt)
    (check-equal? (list (flat (grad p32)) (flat (grad p64))) '((1.0 -0.5) (3.0 4.0)))
    (grad-scaler-step! s opt)
    (grad-scaler-update! s)
    (check-equal? (flat p64) (list (- 1.5 (* 0.1 3.0)) (- 2.0 (* 0.1 4.0))))))
