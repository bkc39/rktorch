#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" * backward! sum tensor tensor->list)
           (only-in "../nn.rkt"
                    Parameter grad-scaler grad-scaler-scale grad-scaler-scale-loss
                    grad-scaler-step! grad-scaler-update! sgd zero-grads!)
           (only-in "private/python-env.rkt" python-check python-torch-available? tol))

  (define inf-steps '(2 5))

  (define (coefficients t)
    (list (if (memv t inf-steps) +inf.0 (+ 0.5 (* 0.1 t)))
          (- 0.25 (* 0.05 t))))

  (define (scripted-run)
    (define p (Parameter (tensor '(1.0 -1.0))))
    (define opt (sgd (list p) #:lr 0.1 #:momentum 0.9))
    (define s (grad-scaler #:device 'cpu #:init-scale 1024.0 #:growth-interval 2))
    (define scales
      (for/list ([t (in-range 8)])
        (zero-grads! opt)
        (backward! (grad-scaler-scale-loss s (sum (* p (tensor (coefficients t))))))
        (grad-scaler-step! s opt)
        (grad-scaler-update! s)
        (grad-scaler-scale s)))
    (values scales (tensor->list p)))

  (cond
    [(not (python-torch-available?))
     (displayln "[grad-scaler-parity-test] skipped: python3 `torch` not available")]
    [else
     (define j (python-check "grad_scaler.py"))
     (define-values (scales params) (scripted-run))
     (check-equal? scales (hash-ref j 'scales) "G3-L4: the scale after each update")
     (for ([r (in-list params)] [w (in-list (hash-ref j 'params))] [i (in-naturals)])
       (check-= r w tol (format "G3-L4: parameter ~a" i)))]))
