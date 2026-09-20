#lang racket/base

(require (only-in racket/contract/base ->* list/c)
         (only-in threading ~>)
         (only-in "../foreign.rkt" adaptive-avg-pool2d add flatten relu)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../nn/batch-norm.rkt" BatchNorm2d)
         (only-in "../nn/conv.rkt" Conv2d)
         (only-in "../nn/layer.rkt" define-layer)
         (only-in "../nn/linear.rkt" Linear)
         (only-in "../nn/sequential.rkt" Sequential))

(define-layer BasicBlock (conv1 bn1 conv2 bn2 shortcut) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:stride exact-positive-integer?]
                  basic-block?)
  #:init (in out #:stride [stride 1])
  (set! conv1 (Conv2d in out 3 #:stride stride #:padding 1 #:bias? #f))
  (set! bn1 (BatchNorm2d out))
  (set! conv2 (Conv2d out out 3 #:padding 1 #:bias? #f))
  (set! bn2 (BatchNorm2d out))
  (set! shortcut
        (and (or (not (= stride 1)) (not (= in out)))
             (Sequential (Conv2d in out 1 #:stride stride #:bias? #f)
                         (BatchNorm2d out))))
  #:forward ([x : image-batch/c])
  (relu (add (bn2 (conv2 (relu (bn1 (conv1 x)))))
             (if shortcut (shortcut x) x))))

(define four-stage-depths/c
  (list/c exact-positive-integer? exact-positive-integer?
          exact-positive-integer? exact-positive-integer?))

(define (stage in out blocks stride)
  (apply Sequential
         (BasicBlock in out #:stride stride)
         (for/list ([_ (in-range (sub1 blocks))])
           (BasicBlock out out))))

(define-layer ResNet (stem bn layer1 layer2 layer3 layer4 fc) ;; noqa
  #:predicate resnet?
  #:contract (->* []
                  [#:classes exact-positive-integer?
                   #:base exact-positive-integer?
                   #:blocks four-stage-depths/c]
                  resnet?)
  #:init (#:classes [classes 10] #:base [base 64] #:blocks [blocks '(2 2 2 2)])
  (set! stem (Conv2d 3 base 3 #:padding 1 #:bias? #f))
  (set! bn (BatchNorm2d base))
  (set! layer1 (stage base base (list-ref blocks 0) 1))
  (set! layer2 (stage base (* 2 base) (list-ref blocks 1) 2))
  (set! layer3 (stage (* 2 base) (* 4 base) (list-ref blocks 2) 2))
  (set! layer4 (stage (* 4 base) (* 8 base) (list-ref blocks 3) 2))
  (set! fc (Linear (* 8 base) classes))
  #:forward ([x : image-batch/c])
  (~> x stem bn relu layer1 layer2 layer3 layer4
      (adaptive-avg-pool2d 1) (flatten 1) fc))
