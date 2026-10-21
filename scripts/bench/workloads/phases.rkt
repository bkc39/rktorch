#lang racket/base

(require (only-in torch
                  backward! [device device-of] device-type item make-generator manual-seed! randn tensor
                  with-autocast with-default-device)
         (only-in torch/data/loader dataloader in-dataloader tensor-dataset)
         (only-in torch/nn adam cross-entropy one-cycle-lr parameters sgd step! zero-grads!)
         (only-in torch/private/step-log settle!)
         (only-in torch/scribblings/examples/05-mnist convnet)
         (only-in torch/vision/resnet ResNet)
         (only-in torch/vision/transforms random-crop random-horizontal-flip))

(provide (struct-out phased)
         mnist-phases
         resnet-phases
         run-phases!)

(struct phased (example device settings phases sync!))

(define (batches loader)
  (define (restart) (call-with-values (lambda () (sequence-generate (in-dataloader loader))) cons))
  (define source (box (restart)))
  (lambda ()
    (unless ((car (unbox source)))
      (set-box! source (restart)))
    ((cdr (unbox source)))))

(define (labels n)
  (tensor (for/list ([i (in-range n)]) (modulo i 10)) #:dtype 'int64))

(define (synthetic-loader dims samples batch generator)
  (dataloader (tensor-dataset (apply randn samples dims) (labels samples))
              #:batch-size batch #:shuffle? #t #:generator generator))

(define (mnist-phases device #:batch [batch 128] #:samples [samples 1024])
  (with-default-device device
    (manual-seed! 0)
    (define next-batch
      (batches (synthetic-loader '(1 28 28) samples batch (make-generator 0))))
    (define net (convnet))
    (define opt (adam (parameters net) #:lr 0.001))
    (define xb #f)
    (define yb #f)
    (define loss #f)
    (phased "05-mnist" device (hasheq 'batch batch)
            (list (cons 'data (lambda () (set!-values (xb yb) (next-batch))))
                  (cons 'zero-grads (lambda () (zero-grads! opt)))
                  (cons 'forward (lambda () (set! loss (cross-entropy (net xb) yb))))
                  (cons 'backward (lambda () (backward! loss)))
                  (cons 'optimizer (lambda () (step! opt))))
            (lambda () (settle! (parameters net))))))

(define (resnet-phases device #:batch [batch 128] #:samples [samples 1024] #:base [base 64])
  (with-default-device device
    (manual-seed! 0)
    (define g (make-generator 0))
    (define next-batch (batches (synthetic-loader '(3 32 32) samples batch g)))
    (define net (ResNet #:base base))
    (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9 #:nesterov? #t #:weight-decay 5e-4))
    (define schedule (one-cycle-lr opt #:max-lr 0.1 #:total-steps 1000000))
    (define (forward xs ys)
      (define batch-device (device-of xs))
      (if (eq? (device-type batch-device) 'cuda)
          (with-autocast #:device batch-device (cross-entropy (net xs) ys))
          (cross-entropy (net xs) ys)))
    (define xb #f)
    (define yb #f)
    (define loss #f)
    (phased "09-resnet" device (hasheq 'batch batch 'base base)
            (list (cons 'data
                        (lambda ()
                          (define-values (x y) (next-batch))
                          (set! xb (random-horizontal-flip (random-crop x #:padding 4 #:generator g)
                                                           #:generator g))
                          (set! yb y)))
                  (cons 'zero-grads (lambda () (zero-grads! opt)))
                  (cons 'forward (lambda () (set! loss (forward xb yb))))
                  (cons 'backward (lambda () (backward! loss)))
                  (cons 'optimizer (lambda () (step! opt) (step! schedule)))
                  (cons 'loss-item (lambda () (item loss))))
            (lambda () (settle! (parameters net))))))

(define (run-phases! p
                     #:around [around (lambda (_name thunk) (thunk))]
                     #:skip [skip '()])
  (with-default-device (phased-device p)
    (for ([ph (in-list (phased-phases p))] #:unless (memq (car ph) skip))
      (around (car ph) (cdr ph)))))
