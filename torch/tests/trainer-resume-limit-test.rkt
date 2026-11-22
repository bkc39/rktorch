#lang racket/base

(module+ test
  (require rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" saving-callback Small)
           (only-in "private/trainer-kit.rkt" toy-loader))

  (define (epoch-2-checkpoint dir)
    (define path (build-path dir "end.ckpt"))
    (void (fit (Small) (toy-loader 4) #:val (toy-loader 2) #:max-epochs 2 #:logger #f
               #:callbacks (list (saving-callback (lambda (_r) path)))))
    path)

  (test-case "T3-L5: resuming at the limit runs no batch, and past it raises"
    (with-temporary-directory (dir #:template "limit-~a")
      (define path (epoch-2-checkpoint dir))
      (define seen (box '()))
      (define ((note! name) _r . _)
        (set-box! seen (cons name (unbox seen))))
      (void (fit (Small) (toy-loader 4) #:max-epochs 2 #:logger #f #:ckpt-path path
                 #:callbacks (list (lambda-callback #:on-train-start (note! 'start)
                                                    #:on-train-batch-start (note! 'batch)
                                                    #:on-train-end (note! 'end)))))
      (check-equal? (reverse (unbox seen)) '(start end))
      (check-exn #rx"current_epoch=2.*max_epochs=1|epoch 2.*#:max-epochs 1"
                 (lambda () (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f
                                 #:ckpt-path path))))))
