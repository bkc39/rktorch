#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * backward! grad item manual-seed! randn reshape sum tensor tensor->list)
           (only-in "../nn.rkt"
                    Linear Parameter Sequential adam grad-scaler grad-scaler-enabled?
                    grad-scaler-scale grad-scaler-scale-loss grad-scaler-step!
                    grad-scaler-unscale! grad-scaler-update! layer-copy parameters sgd step!
                    zero-grads!)
           (only-in (submod "../nn/optim.rkt" private) optimizer-state)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/op-calls.rkt" crossings-of)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "grad-scaler-test"))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (cpu-scaler)
    (grad-scaler #:device 'cpu #:growth-interval 3))

  (define (gradients ps)
    (map (lambda (p) (flat (grad p))) ps))

  (bounded-test "G3-L1: scaling a loss multiplies it by the scale exactly"
    (define s (cpu-scaler))
    (define loss (tensor 0.3))
    (check-equal? (item (grad-scaler-scale-loss s loss)) (* 65536.0 (item loss))))

  (bounded-test "G3-L2: unscaled gradients of a scaled loss are the unscaled run's bits"
    (manual-seed! 0)
    (define net (Sequential (Linear 4 8) (Linear 8 2)))
    (define twin (layer-copy net))
    (define x (randn 5 4))
    (define s (cpu-scaler))
    (define opt (sgd (parameters net) #:lr 0.1))
    (backward! (grad-scaler-scale-loss s (sum (net x))))
    (grad-scaler-unscale! s opt)
    (~> (twin x) sum backward!)
    (check-equal? (gradients (parameters net)) (gradients (parameters twin))))

  (define (inf-step! s opt p bad?)
    (zero-grads! opt)
    (define c (tensor (if bad? '(1.0 +inf.0) '(1.0 0.5))))
    (backward! (grad-scaler-scale-loss s (sum (* p c))))
    (grad-scaler-step! s opt)
    (grad-scaler-update! s))

  (bounded-test "G3-L3: an inf gradient skips the step and halves the scale"
    (define p (Parameter (tensor '(0.5 -0.25))))
    (define opt (adam (list p) #:lr 0.1))
    (define s (cpu-scaler))
    (inf-step! s opt p #f)
    (define params-before (flat p))
    (define state-before (map flat (optimizer-state opt)))
    (inf-step! s opt p #t)
    (check-equal? (flat p) params-before)
    (check-equal? (map flat (optimizer-state opt)) state-before)
    (check-equal? (grad-scaler-scale s) 32768.0)
    (for ([_ (in-range 3)]) (inf-step! s opt p #f))
    (check-equal? (grad-scaler-scale s) 65536.0))

  (define (scaled-setup)
    (define p (Parameter (tensor '(0.5 -0.25))))
    (define opt (sgd (list p) #:lr 0.1))
    (define s (cpu-scaler))
    (backward! (grad-scaler-scale-loss s (sum (* p p))))
    (values s opt))

  (define protocol-errors
    `(("unscale before any loss is scaled"
       ,(lambda () (grad-scaler-unscale! (cpu-scaler) (sgd (list (Parameter (tensor 1.0)))
                                                            #:lr 0.1))))
      ("unscale twice"
       ,(lambda ()
          (define-values (s opt) (scaled-setup))
          (grad-scaler-unscale! s opt)
          (grad-scaler-unscale! s opt)))
      ("unscale after step"
       ,(lambda ()
          (define-values (s opt) (scaled-setup))
          (grad-scaler-step! s opt)
          (grad-scaler-unscale! s opt)))
      ("step twice"
       ,(lambda ()
          (define-values (s opt) (scaled-setup))
          (grad-scaler-step! s opt)
          (grad-scaler-step! s opt)))
      ("update without inf checks"
       ,(lambda ()
          (define-values (s _opt) (scaled-setup))
          (grad-scaler-update! s)))))

  (for ([case (in-list protocol-errors)])
    (match-define (list label call) case)
    (bounded-test (format "G3-L5: ~a blames the caller" label)
      (check-exn blames-this-test? call)))

  (bounded-test "G3-L6: a disabled scaler passes the loss through and steps as step!"
    (manual-seed! 0)
    (define net (Linear 3 2))
    (define twin (layer-copy net))
    (define s (grad-scaler #:device 'cpu #:enabled? #f))
    (define opt (sgd (parameters net) #:lr 0.1))
    (define twin-opt (sgd (parameters twin) #:lr 0.1))
    (define x (randn 4 3))
    (define loss (sum (net x)))
    (check-eq? (grad-scaler-scale-loss s loss) loss)
    (backward! loss)
    (grad-scaler-step! s opt)
    (grad-scaler-update! s)
    (~> (twin x) sum backward!)
    (step! twin-opt)
    (check-equal? (map flat (parameters net)) (map flat (parameters twin)))
    (check-false (grad-scaler-enabled? s))
    (check-equal? (grad-scaler-scale s) 1.0))

  (define (step-crossings n #:fused? fused? #:scaled? scaled?)
    (manual-seed! 0)
    (define net (apply Sequential (for/list ([_ (in-range n)]) (Linear 3 3))))
    (define opt (if fused? (adam (parameters net) #:fused? #t) (adam (parameters net))))
    (define s (cpu-scaler))
    (define x (randn 2 3))
    (for/last ([_ (in-range 4)])
      (zero-grads! opt)
      (define loss (sum (net x)))
      (cond
        [scaled?
         (backward! (grad-scaler-scale-loss s loss))
         (begin0 (crossings-of (lambda () (grad-scaler-step! s opt)))
                 (grad-scaler-update! s))]
        [else
         (backward! loss)
         (crossings-of (lambda () (step! opt)))])))

  (bounded-test "G3-L7: a fused scaled step costs a fused step, a foreach one a constant more"
    (check-equal? (step-crossings 2 #:fused? #t #:scaled? #t)
                  (step-crossings 2 #:fused? #t #:scaled? #f))
    (define (extra n)
      (- (step-crossings n #:fused? #f #:scaled? #t) (step-crossings n #:fused? #f #:scaled? #f)))
    (check-equal? (extra 1) (extra 4))
    (check-true (positive? (extra 1)))))
