#lang racket/base

(require (only-in racket/contract/base ->*)
         (only-in "../foreign.rkt"
                  add batch-norm copy! ones tensor to-dtype zeros)
         (only-in "../foreign/contracts.rkt" feature-batch/c image-batch/c)
         (only-in "buffer.rkt" Buffer)
         (only-in "layer.rkt" define-layer training? with-mode)
         (only-in "parameter.rkt" Parameter))

(define (normalize x mode momentum eps weight bias
                   running-mean running-var num-batches-tracked)
  (define training (training? mode))
  (when training
    ;; nn.BatchNorm2d counts the batches it has normalised; add promotes the
    ;; int64 counter, so the sum is cast back before it lands in the buffer
    (copy! num-batches-tracked
           (to-dtype (add num-batches-tracked 1) 'int64)))
  (batch-norm x
              #:running-mean running-mean
              #:running-var running-var
              #:weight weight
              #:bias bias
              #:training? training
              #:momentum momentum
              #:eps eps))

(define-layer BatchNorm2d (eps momentum weight bias ;; noqa
                           running-mean running-var num-batches-tracked)
  #:contract (->* [exact-positive-integer?]
                  [#:eps real? #:momentum real?]
                  batch-norm2d?)
  #:init (num-features #:eps [eps 1e-5] #:momentum [momentum 0.1])
  (set! weight (Parameter (ones num-features)))
  (set! bias (Parameter (zeros num-features)))
  (set! running-mean (Buffer (zeros num-features)))
  (set! running-var (Buffer (ones num-features)))
  (set! num-batches-tracked (Buffer (tensor 0)))
  #:forward ([x : image-batch/c])
  (with-mode
    (normalize x mode momentum eps weight bias
               running-mean running-var num-batches-tracked)))

(define-layer BatchNorm1d (eps momentum weight bias ;; noqa
                           running-mean running-var num-batches-tracked)
  #:contract (->* [exact-positive-integer?]
                  [#:eps real? #:momentum real?]
                  batch-norm1d?)
  #:init (num-features #:eps [eps 1e-5] #:momentum [momentum 0.1])
  (set! weight (Parameter (ones num-features)))
  (set! bias (Parameter (zeros num-features)))
  (set! running-mean (Buffer (zeros num-features)))
  (set! running-var (Buffer (ones num-features)))
  (set! num-batches-tracked (Buffer (tensor 0)))
  #:forward ([x : feature-batch/c])
  (with-mode
    (normalize x mode momentum eps weight bias
               running-mean running-var num-batches-tracked)))
