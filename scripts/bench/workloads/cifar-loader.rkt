#lang racket/base

(module+ main
  (require (only-in torch accelerator-if-available make-generator manual-seed! to)
           (only-in torch/data/loader in-dataloader)
           (only-in "../../../examples/test/private/env.rkt" env-number)
           (only-in "cifar-pipeline.rkt" cifar-folder-loader cifar-folder-root log-batch!))

  (define epochs (env-number "EPOCHS" 2))
  (define workers (env-number "WORKERS" 0))
  (define batch (env-number "BATCH" 128))
  (define pipeline (or (getenv "PIPELINE") "per-sample"))
  (define root (cifar-folder-root (or (getenv "DATASET") "full") (getenv "PNG_DIGEST")))
  (define device (accelerator-if-available))

  (manual-seed! 0)
  (define loader
    (cifar-folder-loader root pipeline (make-generator 0)
                         #:batch batch #:workers workers #:drop-last? #t))
  (for* ([_ (in-range epochs)] [(xb yb) (in-dataloader loader)])
    (define x (to xb device))
    (to yb device)
    (log-batch! x)))
