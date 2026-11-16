#lang racket/base

(require (only-in racket/list fourth last)
         (only-in racket/match match-define)
         (only-in torch cpu-device cuda-available? cuda-device device-type)
         (only-in torch/data/loader tensor-dataset)
         (only-in torch/scribblings/examples/20-fashion-mnist fashion-recipe train-fashion-mnist)
         (only-in torch/tests/private/python-env call-with-python-env python-result)
         (only-in torch/train lambda-callback run-epoch run-estimated-stepping-batches run-metric)
         (only-in torch/vision/fashion-mnist fashion-mnist-dataset load-fashion-mnist-fixture)
         (only-in (submod torch/vision/fashion-mnist private) fashion-mnist-cache-dir))

(provide fixture-batch-size
         full-batch-size
         lightning-version
         device-name
         precision-for
         headline-device
         epoch-recorder
         print-epoch-rows
         run-example
         run-headline
         twin-config
         twin-json)

(define fixture-batch-size 32)
(define full-batch-size 128)
(define lightning-version "2.6.5")

(define (device-name device)
  (symbol->string (device-type device)))

(define (precision-for device)
  (if (eq? (device-type device) 'cuda) '16-mixed '32-true))

(define (headline-device)
  (if (cuda-available?) (cuda-device) (cpu-device)))

(define (epoch-recorder)
  (define rows '())
  (define clock #f)
  (define (start! _r)
    (set! clock (current-inexact-milliseconds)))
  (define (record! r)
    (set! rows (cons (list (add1 (run-epoch r)) (run-metric r 'train/loss)
                           (run-metric r 'val/loss) (run-metric r 'val/acc)
                           (/ (- (current-inexact-milliseconds) clock) 1000.0))
                     rows)))
  (values (lambda-callback #:on-train-epoch-start start! #:on-train-epoch-end record!)
          (lambda () (reverse rows))))

(define (print-epoch-rows rows)
  (for ([row (in-list rows)])
    (match-define (list epoch train-loss val-loss acc seconds) row)
    (printf "epoch ~a: loss ~a, val loss ~a, val acc ~a, ~as\n" epoch
            (real->decimal-string train-loss 4) (real->decimal-string val-loss 4)
            (real->decimal-string acc 4) (real->decimal-string seconds 1)))
  (printf "test acc ~a\n" (real->decimal-string (fourth (last rows)) 4)))

(define (stepping-recorder)
  (define total (box #f))
  (values (lambda-callback
           #:on-train-start (lambda (r) (set-box! total (run-estimated-stepping-batches r))))
          (lambda () (unbox total))))

(define (run-example #:device [device (cpu-device)] #:precision [precision '32-true]
                     #:epochs [epochs 2] #:callbacks [callbacks '()] #:logger [logger #f])
  (define-values (xs ys) (load-fashion-mnist-fixture 'train))
  (define-values (test-xs test-ys) (load-fashion-mnist-fixture 'test))
  (define-values (stepping total) (stepping-recorder))
  (define-values (run classifier)
    (train-fashion-mnist (tensor-dataset xs ys) (tensor-dataset test-xs test-ys)
                         #:batch-size fixture-batch-size #:test-batch-size 64
                         #:epochs epochs #:device device #:precision precision
                         #:log-every-n-steps 1 #:logger logger
                         #:callbacks (cons stepping callbacks)))
  (values run classifier (total)))

(define (run-headline #:epochs epochs #:device device)
  (define-values (recorder rows) (epoch-recorder))
  (define-values (stepping total) (stepping-recorder))
  (define-values (run _classifier)
    (train-fashion-mnist (fashion-mnist-dataset 'train #:device device)
                         (fashion-mnist-dataset 'test #:device device)
                         #:batch-size full-batch-size #:test-batch-size 1000
                         #:epochs epochs #:device device #:precision (precision-for device)
                         #:callbacks (list recorder stepping)))
  (values (rows) run (total)))

(define (twin-config #:batch batch #:epochs epochs #:total total #:warmup warmup)
  (hash-set* fashion-recipe 'batch batch 'epochs epochs 'total total 'warmup warmup
             'lightning lightning-version))

(define (twin-json device precision mode epochs)
  (call-with-python-env
   #:env `(("RKTORCH_PARITY_DEVICE" . ,(device-name device))
           ("RKTORCH_PARITY_PRECISION" . ,(symbol->string precision))
           ("RKTORCH_TWIN_MODE" . ,(symbol->string mode))
           ("EPOCHS" . ,(number->string epochs))
           ("RKTORCH_FASHION_MNIST_DIR" . ,(path->string (fashion-mnist-cache-dir))))
   (lambda () (python-result "python/20_fashion_mnist.py"))))
