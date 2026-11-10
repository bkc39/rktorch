#lang racket/base

(require (for-syntax racket/base syntax/parse/pre)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../main.rkt"
                  * - ~> adaptive-avg-pool2d backward! conv2d embedding
                  cudnn-benchmark? cudnn-deterministic?
                  deterministic-algorithms-enabled?
                  deterministic-algorithms-warn-only? grid-sample index-add!
                  index-select log-softmax rand randn scaled-dot-product-attention
                  scatter-add! set-cudnn-benchmark! set-cudnn-deterministic! sum
                  tensor upsample-bicubic2d use-deterministic-algorithms! zeros)
         (only-in "../../nn.rkt" ctc-loss nll-loss))

(provide call-with-switches-restored
         default-switches
         determinism-rows
         set-switches!
         switch-values
         with-switches-restored)

(define (switch-values)
  (hash 'deterministic-algorithms (deterministic-algorithms-enabled?)
        'warn-only (deterministic-algorithms-warn-only?)
        'cudnn-deterministic (cudnn-deterministic?)
        'cudnn-benchmark (cudnn-benchmark?)))

(define default-switches
  (hash 'deterministic-algorithms #f
        'warn-only #f
        'cudnn-deterministic #f
        'cudnn-benchmark #f))

(define (set-switches! switches)
  (use-deterministic-algorithms! (hash-ref switches 'deterministic-algorithms)
                                 #:warn-only? (hash-ref switches 'warn-only))
  (set-cudnn-deterministic! (hash-ref switches 'cudnn-deterministic))
  (set-cudnn-benchmark! (hash-ref switches 'cudnn-benchmark)))

(define (call-with-switches-restored thunk)
  (define saved (switch-values))
  (dynamic-wind void thunk (lambda () (set-switches! saved))))

(define-syntax-parse-rule (with-switches-restored body:expr ...+)
  (call-with-switches-restored (lambda () body ...)))

(define (bins on)
  (tensor '(0 3 1 3 7 2 3 0 5 3 6 1 4 3 2 7) #:dtype 'int64 #:device on))

(define (leaf on . dims)
  (apply randn dims #:device on #:requires-grad? #t))

(define determinism-rows
  (list
   (cons 'adaptive-avg-pool2d-backward
         (lambda (on)
           (~> (leaf on 1 2 5 5) (adaptive-avg-pool2d '(3 3)) sum backward!)))
   (cons 'ctc-loss-backward
         (lambda (on)
           (~> (leaf on 10 2 5)
               (log-softmax 2)
               (ctc-loss (tensor '((1 2 3 4) (2 3 4 1)) #:dtype 'int64
                                 #:device on)
                         #:input-lengths '(10 10)
                         #:target-lengths '(4 4))
               backward!)))
   (cons 'grid-sample-backward
         (lambda (on)
           (~> (leaf on 1 1 4 4)
               (grid-sample (~> (rand 1 3 3 2 #:device on) (* 1.8) (- 0.9)))
               sum
               backward!)))
   (cons 'upsample-bicubic2d-backward
         (lambda (on)
           (~> (leaf on 1 1 4 4) (upsample-bicubic2d '(8 8)) sum backward!)))
   (cons 'index-add
         (lambda (on)
           (index-add! (zeros 8 #:device on) 0 (bins on) (randn 16 #:device on))))
   (cons 'scatter-add
         (lambda (on)
           (scatter-add! (zeros 8 #:device on) 0 (bins on)
                         (randn 16 #:device on))))
   (cons 'index-select-backward
         (lambda (on)
           (~> (leaf on 8 4) (index-select 0 (bins on)) sum backward!)))
   (cons 'embedding-backward
         (lambda (on)
           (~> (embedding (bins on) (leaf on 8 4)) sum backward!)))
   (cons 'conv2d-backward
         (lambda (on)
           (~> (randn 1 1 8 8 #:device on) (conv2d (leaf on 2 1 3 3)) sum
               backward!)))
   (cons 'nll-loss-forward
         (lambda (on)
           (nll-loss (log-softmax (randn 4 5 #:device on) 1)
                     (tensor '(0 1 2 3) #:dtype 'int64 #:device on))))
   (cons 'scaled-dot-product-attention-backward
         (lambda (on)
           (~> (scaled-dot-product-attention (leaf on 1 2 8 16)
                                             (leaf on 1 2 8 16)
                                             (leaf on 1 2 8 16))
               sum
               backward!)))))
