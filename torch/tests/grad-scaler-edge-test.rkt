#lang racket/base

(module+ test
  (require (only-in racket/match match match-define struct*)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * + backward! cuda-available? grad manual-seed! randn sum tensor)
           (only-in "../nn.rkt"
                    Linear Parameter adam grad-scaler grad-scaler-enabled? grad-scaler-scale
                    grad-scaler-scale-loss grad-scaler-step! grad-scaler-unscale!
                    grad-scaler-update! layer-copy parameters sgd step! zero-grads!)
           (only-in (submod "../nn/optim.rkt" private) optimizer-state)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/flat.rkt" flat)
           (only-in "private/plain-sgd.rkt" plain-sgd)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "grad-scaler-edge-test"))

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

  (define (fused-adam ps)
    (adam ps #:fused? #t #:lr 0.1))

  (define (fused-sgd ps)
    (sgd ps #:fused? #t #:lr 0.1 #:momentum 0.9))

  (define (fused-run mode bad? #:optimizer [make-opt fused-adam])
    (manual-seed! 0)
    (define net (Linear 3 2))
    (define x (randn 4 3))
    (define opt (make-opt (parameters net)))
    (define loss (* (sum (net x)) (if bad? +inf.0 1.0)))
    (define s (cpu-scaler))
    (match mode
      ['plain
       (backward! loss)
       (step! opt)]
      [_
       (backward! (grad-scaler-scale-loss s loss))
       (when (eq? mode 'unscaled)
         (grad-scaler-unscale! s opt))
       (grad-scaler-step! s opt)
       (grad-scaler-update! s)])
    (map flat (parameters net)))

  (bounded-test "a fused step after an explicit unscale equals the plain fused step"
    (check-equal? (fused-run 'unscaled #f) (fused-run 'plain #f)))

  (bounded-test "a fused step after an explicit unscale skips an inf step"
    (manual-seed! 0)
    (define before (map flat (parameters (Linear 3 2))))
    (check-equal? (fused-run 'unscaled #t) before))

  (bounded-test "a fused step with no explicit unscale equals the plain fused step"
    (check-equal? (fused-run 'ready #f) (fused-run 'plain #f))
    (check-equal? (fused-run 'ready #f #:optimizer fused-sgd)
                  (fused-run 'plain #f #:optimizer fused-sgd)))

  (bounded-test "a fused step with no explicit unscale leaves its state on an inf step"
    (manual-seed! 0)
    (define net (Linear 3 2))
    (define x (randn 4 3))
    (define opt (adam (parameters net) #:fused? #t #:lr 0.1))
    (define s (cpu-scaler))
    (define (scaled-step! factor)
      (zero-grads! opt)
      (define loss (~> (net x) sum (* factor)))
      (backward! (grad-scaler-scale-loss s loss))
      (grad-scaler-step! s opt)
      (grad-scaler-update! s))
    (scaled-step! 1.0)
    (define params-before (map flat (parameters net)))
    (define state-before (map flat (optimizer-state opt)))
    (scaled-step! +inf.0)
    (check-equal? (map flat (parameters net)) params-before)
    (check-equal? (map flat (optimizer-state opt)) state-before)
    (check-equal? (grad-scaler-scale s) 32768.0))

  (bounded-test "an update to a given scale needs no step"
    (define p (Parameter (tensor '(0.5 -0.25))))
    (define s (cpu-scaler))
    (grad-scaler-scale-loss s (sum (* p p)))
    (grad-scaler-update! s 8.0)
    (check-equal? (grad-scaler-scale s) 8.0))

  (bounded-test "an update before any loss is scaled blames the caller"
    (check-exn blames-this-test? (lambda () (grad-scaler-update! (cpu-scaler) 8.0))))

  (bounded-test "unscaling float16 gradients raises, as torch's unscale_ does"
    (define p (Parameter (tensor '(0.5 -0.25) #:dtype 'float16)))
    (define opt (sgd (list p) #:lr 0.1))
    (define s (cpu-scaler))
    (backward! (grad-scaler-scale-loss s (sum (* p p))))
    (check-exn #rx"grad-scaler-unscale!: attempting to unscale FP16 gradients"
               (lambda () (grad-scaler-unscale! s opt)))
    (check-exn #rx"grad-scaler-step!: attempting to unscale FP16 gradients"
               (lambda () (grad-scaler-step! s opt))))

  (bounded-test "a float16 gradient raises before any other gradient is unscaled"
    (define p32 (Parameter (tensor '(0.5 -0.25))))
    (define p16 (Parameter (tensor '(0.5 -0.25) #:dtype 'float16)))
    (define opt (sgd (list p32 p16) #:lr 0.1))
    (define s (cpu-scaler))
    (backward! (grad-scaler-scale-loss s (+ (sum (* p32 p32)) (sum (* p16 p16)))))
    (define scaled (flat (grad p32)))
    (check-exn #rx"attempting to unscale FP16 gradients"
               (lambda () (grad-scaler-unscale! s opt)))
    (check-equal? (flat (grad p32)) scaled)
    (check-equal? scaled '(65536.0 -32768.0)))

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
    (check-equal? (flat p64) (list (- 1.5 (* 0.1 3.0)) (- 2.0 (* 0.1 4.0)))))

  (struct plain-run (unscaled twin-grads before stepped params scale))

  (define (plain-scaled-run bad?)
    (manual-seed! 0)
    (define net (Linear 3 2))
    (define twin (layer-copy net))
    (define x (randn 4 3))
    (define opt (plain-sgd (parameters net) 0.1))
    (define s (cpu-scaler))
    (backward! (grad-scaler-scale-loss s (* (sum (net x)) (if bad? +inf.0 1.0))))
    (grad-scaler-unscale! s opt)
    (define unscaled (map (lambda (p) (flat (grad p))) (parameters net)))
    (grad-scaler-step! s opt)
    (grad-scaler-update! s)
    (~> (twin x) sum backward!)
    (define twin-grads (map (lambda (p) (flat (grad p))) (parameters twin)))
    (define before (map flat (parameters twin)))
    (~> (parameters twin) (plain-sgd 0.1) step!)
    (plain-run unscaled twin-grads before (map flat (parameters twin))
               (map flat (parameters net)) (grad-scaler-scale s)))

  (bounded-test "a user gen:optimizer unscales and steps as the unscaled run"
    (match-define (struct* plain-run ([unscaled unscaled] [twin-grads twin-grads] [stepped stepped]
                                      [params params] [scale scale]))
      (plain-scaled-run #f))
    (check-equal? unscaled twin-grads)
    (check-equal? params stepped)
    (check-equal? scale 65536.0))

  (bounded-test "a user gen:optimizer skips an inf step and halves the scale"
    (match-define (struct* plain-run ([before before] [params params] [scale scale]))
      (plain-scaled-run #t))
    (check-equal? params before)
    (check-equal? scale 32768.0)))
