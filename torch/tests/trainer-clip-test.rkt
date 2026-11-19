#lang racket/base

(module+ test
  (require (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in threading ~>)
           (only-in "../data/loader.rkt" dataloader in-dataloader tensor-dataset)
           (only-in "../main.rkt"
                    backward! cat grad item manual-seed! randn reshape tensor->list)
           (only-in "../nn.rkt"
                    Linear clip-grad-norm! clip-grad-value! layer-copy mse-loss parameters sgd
                    step! zero-grads!)
           "../train.rkt"
           (only-in "private/python-env.rkt"
                    call-with-python-env python-result python-torch-available? tol)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path twin "python/trainer_twin.py")

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (gradients net)
    (map (lambda (p) (flat (grad p))) (parameters net)))

  (define-trainer Clipped (net seen)
    (define-init (model sink) (set! net model) (set! seen sink))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-before-optimizer-step optimizer)
      (set-box! seen (cons (gradients net) (unbox seen)))))

  (define (loader)
    (manual-seed! 1)
    (dataloader (tensor-dataset (randn 12 4) (randn 12 2)) #:batch-size 4))

  (define (clipper algorithm)
    (if (eq? algorithm 'norm)
        (lambda (ps) (clip-grad-norm! ps 0.05))
        (lambda (ps) (clip-grad-value! ps 0.01))))

  (define (hand-loop net algorithm)
    (define opt (sgd (parameters net) #:lr 0.1))
    (define clip! (clipper algorithm))
    (define seen
      (for*/list ([_ (in-range 2)] [(x y) (in-dataloader (loader))])
        (zero-grads! opt)
        (~> (net x) (mse-loss y) backward!)
        (begin0 (gradients net)
                (clip! (parameters net))
                (step! opt))))
    (values seen (map flat (parameters net))))

  (for ([algorithm (in-list '(norm value))])
    (bounded-test (format "G3-L13: clip algorithm '~a clips after the hook, before the step"
                          algorithm)
      (manual-seed! 0)
      (define net (Linear 4 2))
      (define twin-net (layer-copy net))
      (define seen (box '()))
      (fit (Clipped net seen) (loader) #:max-epochs 2 #:logger #f
           #:gradient-clip-val (if (eq? algorithm 'norm) 0.05 0.01)
           #:gradient-clip-algorithm algorithm)
      (define-values (hand-seen hand-params) (hand-loop twin-net algorithm))
      (check-equal? (reverse (unbox seen)) hand-seen)
      (check-equal? (map flat (parameters net)) hand-params)))

  (define-trainer Twin (net losses)
    (define-init (model sink) (set! net model) (set! losses sink))
    (define-optimizers (sgd (parameters net) #:lr 0.05 #:momentum 0.9))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (set-box! losses (cons (item loss) (unbox losses)))
      loss))

  (define (fit-config config)
    (manual-seed! 0)
    (define net (Linear 8 4))
    (define losses (box '()))
    (define data (dataloader (tensor-dataset (randn 32 8) (randn 32 4)) #:batch-size 8))
    (if (eq? config 'clip)
        (fit (Twin net losses) data #:max-epochs 2 #:logger #f #:gradient-clip-val 1.0)
        (fit (Twin net losses) data #:max-epochs 2 #:logger #f #:precision 'bf16-mixed))
    (values (reverse (unbox losses))
            (tensor->list (cat (for/list ([p (in-list (parameters net))]) (reshape p -1))))))

  (define (twin-result config)
    (call-with-python-env
     #:env (list (cons "RKTORCH_PARITY_DEVICE" "cpu")
                 (cons "RKTORCH_TWIN_CONFIG" (symbol->string config)))
     (lambda () (python-result twin))))

  (for ([config (in-list '(clip bf16))])
    (bounded-test (format "G3-L14: the ~a configuration against trainer_twin.py" config)
      (cond
        [(not (python-torch-available?))
         (displayln "[trainer-clip-test] G3-L14 skipped: python3 `torch` not available")]
        [else
         (define j (twin-result config))
         (define-values (losses params) (fit-config config))
         (for ([r (in-list losses)] [p (in-list (hash-ref j 'losses))] [i (in-naturals)])
           (check-= r p tol (format "~a: loss at step ~a" config i)))
         (for ([r (in-list params)] [p (in-list (hash-ref j 'values))] [i (in-naturals)])
           (check-= r p tol (format "~a: parameter ~a" config i)))]))))
