#lang racket/base

(require (only-in racket/contract/base ->*)
         (only-in "../foreign.rkt"
                  addcmul! batch-norm device ones ones-like tensor zeros)
         (only-in (submod "../foreign.rkt" unsafe) to!)
         (only-in "../foreign/contracts.rkt" feature-batch/c image-batch/c)
         (only-in "buffer.rkt" Buffer)
         (only-in "layer.rkt" define-layer training? with-mode)
         (only-in "parameter.rkt" Parameter))

(define (normalize x mode momentum eps weight bias
                   running-mean running-var num-batches-tracked one)
  (define training (training? mode))
  (when training
    ;; int64 add_ refuses the floating alpha a generated scalar carries (#60);
    ;; a scalar one would promote the counter to float32, stuck at 2^24
    (addcmul! num-batches-tracked one one))
  (batch-norm x
              #:running-mean running-mean
              #:running-var running-var
              #:weight weight
              #:bias bias
              #:training? training
              #:momentum momentum
              #:eps eps))

(define-layer BatchNorm2d (eps momentum weight bias ;; noqa
                           running-mean running-var num-batches-tracked one)
  #:contract (->* [exact-positive-integer?]
                  [#:eps real? #:momentum real?]
                  batch-norm2d?)
  #:init (num-features #:eps [eps 1e-5] #:momentum [momentum 0.1])
  (set! weight (Parameter (ones num-features)))
  (set! bias (Parameter (zeros num-features)))
  (set! running-mean (Buffer (zeros num-features)))
  (set! running-var (Buffer (ones num-features)))
  (set! num-batches-tracked (Buffer (tensor 0)))
  (set! one (ones-like num-batches-tracked))
  #:on-move (to! one (device num-batches-tracked))
  #:forward ([x : image-batch/c])
  (with-mode
    (normalize x mode momentum eps weight bias
               running-mean running-var num-batches-tracked one)))

(define-layer BatchNorm1d (eps momentum weight bias ;; noqa
                           running-mean running-var num-batches-tracked one)
  #:contract (->* [exact-positive-integer?]
                  [#:eps real? #:momentum real?]
                  batch-norm1d?)
  #:init (num-features #:eps [eps 1e-5] #:momentum [momentum 0.1])
  (set! weight (Parameter (ones num-features)))
  (set! bias (Parameter (zeros num-features)))
  (set! running-mean (Buffer (zeros num-features)))
  (set! running-var (Buffer (ones num-features)))
  (set! num-batches-tracked (Buffer (tensor 0)))
  (set! one (ones-like num-batches-tracked))
  #:on-move (to! one (device num-batches-tracked))
  #:forward ([x : feature-batch/c])
  (with-mode
    (normalize x mode momentum eps weight bias
               running-mean running-var num-batches-tracked one)))
