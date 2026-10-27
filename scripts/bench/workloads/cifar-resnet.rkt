#lang racket/base

(module+ main
  (require (only-in torch make-generator manual-seed! to with-default-device)
           (only-in torch/data/loader dataloader-length in-dataloader)
           (only-in torch/nn one-cycle-lr parameters sgd)
           (only-in torch/scribblings/examples/09-resnet pick-device train-step)
           (only-in torch/vision/resnet ResNet)
           (only-in "../../../examples/test/private/env.rkt" env-number)
           (only-in "cifar-pipeline.rkt" cifar-folder-loader cifar-folder-root))

  (define epochs (env-number "EPOCHS" 2))
  (define workers (env-number "WORKERS" 0))
  (define batch (env-number "BATCH" 128))
  (define base (env-number "BASE" 64))
  (define pipeline (or (getenv "PIPELINE") "per-sample"))
  (define root (cifar-folder-root (or (getenv "DATASET") "full") (getenv "PNG_DIGEST")))
  (define device (pick-device))
  (define max-lr 0.1)

  (with-default-device device
    (manual-seed! 0)
    (define loader
      (cifar-folder-loader root pipeline (make-generator 0) #:batch batch #:workers workers))
    (define net (ResNet #:base base))
    (define opt (sgd (parameters net) #:lr max-lr #:momentum 0.9 #:nesterov? #t
                     #:weight-decay 5e-4))
    (define schedule
      (one-cycle-lr opt #:max-lr max-lr #:total-steps (* epochs (dataloader-length loader))))
    (for* ([_ (in-range epochs)] [(xb yb) (in-dataloader loader)])
      (train-step net opt (to xb device) (to yb device) #:schedule schedule))))
