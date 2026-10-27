#lang racket/base

(module+ main
  (require (only-in torch accelerator-if-available make-generator manual-seed! to)
           (only-in torch/data/loader in-dataloader)
           (only-in "cifar-pipeline.rkt" cifar-folder-loader cifar-settings log-batch!))

  (define-values (epochs workers batch pipeline root) (cifar-settings))
  (define device (accelerator-if-available))

  (manual-seed! 0)
  (define loader
    (cifar-folder-loader root pipeline (make-generator 0)
                         #:batch batch #:workers workers #:drop-last? #t))
  (for* ([_ (in-range epochs)] [(xb yb) (in-dataloader loader)])
    (define x (to xb device))
    (to yb device)
    (log-batch! x)))
