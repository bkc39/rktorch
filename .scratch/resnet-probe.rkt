#lang racket/base
;; the Racket half of resnet-probe.py: the same recipes, losses printed
(require torch torch/nn
         (only-in torch/vision/cifar10 load-cifar10-fixture)
         (only-in torch/vision/resnet ResNet))

(define (run lr momentum wd)
  (manual-seed! 0)
  (define-values (xs ys) (load-cifar10-fixture))
  (define net (ResNet #:base 16))
  (define opt (sgd (parameters net) #:lr lr #:momentum momentum
                   #:weight-decay wd))
  (for/list ([_ (in-range 5)])
    (zero-grads! opt)
    (define loss (cross-entropy (net xs) ys))
    (backward! loss)
    (step! opt)
    (item loss)))

(for ([recipe (in-list '(("full" 0.05 0.9 5e-4)
                         ("plain" 0.05 0.0 0.0)
                         ("gentle" 0.005 0.9 5e-4)
                         ("momentum_only" 0.05 0.9 0.0)))])
  (printf "~a: ~a\n" (car recipe) (apply run (cdr recipe)))
  (flush-output))
