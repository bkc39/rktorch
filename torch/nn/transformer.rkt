#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i </c >=/c and/c any contract not/c or/c
                  procedure-arity-includes/c)
         (only-in threading ~>)
         (only-in "../foreign.rkt" add copy! gelu relu with-no-grad)
         (only-in "attention.rkt" MultiheadAttention)
         (only-in (submod "attention.rkt" private)
                  attention-mask/c attention-sequence/c attn-mask-shaped
                  padding-shaped rank same-batch wide)
         (only-in "dropout.rkt" Dropout)
         (only-in "init.rkt" call-without-drawing)
         (only-in "layer-list.rkt" LayerList)
         (only-in "layer-norm.rkt" LayerNorm)
         (only-in "layer.rkt" define-layer in-layers layer? parameters)
         (only-in "linear.rkt" Linear))

(define probability/c (and/c real? (>=/c 0) (</c 1)))

(define activation/c
  (or/c 'relu 'gelu 'gelu-tanh (procedure-arity-includes/c 1)))

(define (gelu-tanh x)
  (gelu x #:approximate 'tanh))

(define (activation->procedure activation)
  (case activation
    [(relu) relu]
    [(gelu) gelu]
    [(gelu-tanh) gelu-tanh]
    [else activation]))

(define (layer/c result?)
  (->i ([d-model exact-positive-integer?]
        #:heads [heads exact-positive-integer?])
       (#:ffn-width [ffn-width exact-positive-integer?]
        #:dropout [p probability/c]
        #:activation [activation activation/c]
        #:norm-first? [norm-first? boolean?]
        #:layer-norm-eps [eps (and/c real? positive?)]
        #:batch-first? [batch-first? boolean?]
        #:bias? [bias? boolean?])
       #:pre/name (d-model heads)
       "#:heads divides the model width"
       (zero? (remainder d-model heads))
       [_ result?]))

(define (stack/c element? result?)
  (->* [(and/c (not/c layer?) (-> element?))
        #:layers exact-positive-integer?]
       [#:norm (or/c #f layer?) #:copies? boolean?]
       result?))

(define (checker who call/c)
  (contract call/c void who 'caller who #f))

(define (encoder-call/c d-model heads batch-first?)
  (->i ([src attention-sequence/c]
        [mask (or/c #f attention-mask/c)]
        [key-padding-mask (or/c #f attention-mask/c)]
        [causal? boolean?])
       #:pre/desc (src) (wide "src" src d-model)
       #:pre/desc (src mask)
       (attn-mask-shaped "mask" mask src src heads batch-first?)
       #:pre/desc (src key-padding-mask)
       (padding-shaped "key-padding-mask" key-padding-mask src batch-first?)
       any))

(define (decoder-call/c d-model heads batch-first?)
  (->i ([tgt attention-sequence/c]
        [memory attention-sequence/c]
        [tgt-mask (or/c #f attention-mask/c)]
        [memory-mask (or/c #f attention-mask/c)]
        [tgt-key-padding-mask (or/c #f attention-mask/c)]
        [memory-key-padding-mask (or/c #f attention-mask/c)]
        [tgt-causal? boolean?]
        [memory-causal? boolean?])
       #:pre/desc (tgt) (wide "tgt" tgt d-model)
       #:pre/desc (memory) (wide "memory" memory d-model)
       #:pre/desc (tgt memory)
       (or (= (rank tgt) (rank memory))
           (string-append "tgt and memory are both batched (rank 3)"
                          " or both unbatched (rank 2)"))
       #:pre/desc (tgt memory)
       (same-batch "tgt" tgt "memory" memory batch-first?)
       #:pre/desc (tgt tgt-mask)
       (attn-mask-shaped "tgt-mask" tgt-mask tgt tgt heads batch-first?)
       #:pre/desc (tgt memory memory-mask)
       (attn-mask-shaped "memory-mask" memory-mask tgt memory heads
                         batch-first?)
       #:pre/desc (tgt tgt-key-padding-mask)
       (padding-shaped "tgt-key-padding-mask" tgt-key-padding-mask tgt
                       batch-first?)
       #:pre/desc (tgt memory memory-key-padding-mask)
       (padding-shaped "memory-key-padding-mask" memory-key-padding-mask
                       memory batch-first?)
       any))

(define (residual x norm-first? norm sublayer)
  (if norm-first?
      (add x (sublayer (norm x)))
      (norm (add x (sublayer x)))))

(define (normed norm x)
  (if norm (norm x) x))

(define-layer TransformerEncoderLayer ;; noqa
  (self-attn linear1 dropout linear2 norm1 norm2 dropout1 dropout2
   activation norm-first? check)
  #:contract (layer/c transformer-encoder-layer?)
  #:init (d-model #:heads heads
                  #:ffn-width [ffn-width 2048]
                  #:dropout [p 0.1]
                  #:activation [activation 'relu]
                  #:norm-first? [norm-first? #f]
                  #:layer-norm-eps [eps 1e-5]
                  #:batch-first? [batch-first? #f]
                  #:bias? [bias? #t])
  (set! self-attn (MultiheadAttention d-model #:heads heads #:dropout p
                                      #:bias? bias?
                                      #:batch-first? batch-first?))
  (set! linear1 (Linear d-model ffn-width #:bias? bias?))
  (set! dropout (Dropout #:p p))
  (set! linear2 (Linear ffn-width d-model #:bias? bias?))
  (set! norm1 (LayerNorm d-model #:eps eps #:bias? bias?))
  (set! norm2 (LayerNorm d-model #:eps eps #:bias? bias?))
  (set! dropout1 (Dropout #:p p))
  (set! dropout2 (Dropout #:p p))
  (set! activation (activation->procedure activation))
  (set! check (checker 'TransformerEncoderLayer
                       (encoder-call/c d-model heads batch-first?)))
  #:forward (src
             #:mask [mask #f]
             #:key-padding-mask [padding #f]
             #:causal? [causal? #f])
  (check src mask padding causal?)
  (~> src
      (residual norm-first? norm1
                (lambda (x)
                  (dropout1 (self-attn x x x
                                       #:attn-mask mask
                                       #:key-padding-mask padding
                                       #:causal? causal?))))
      (residual norm-first? norm2
                (lambda (x)
                  (~> x linear1 activation dropout linear2 dropout2)))))

(define-layer TransformerDecoderLayer ;; noqa
  (self-attn multihead-attn linear1 dropout linear2 norm1 norm2 norm3
   dropout1 dropout2 dropout3 activation norm-first? check)
  #:contract (layer/c transformer-decoder-layer?)
  #:init (d-model #:heads heads
                  #:ffn-width [ffn-width 2048]
                  #:dropout [p 0.1]
                  #:activation [activation 'relu]
                  #:norm-first? [norm-first? #f]
                  #:layer-norm-eps [eps 1e-5]
                  #:batch-first? [batch-first? #f]
                  #:bias? [bias? #t])
  (define (attention)
    (MultiheadAttention d-model #:heads heads #:dropout p #:bias? bias?
                        #:batch-first? batch-first?))
  (define (norm)
    (LayerNorm d-model #:eps eps #:bias? bias?))
  (set! self-attn (attention))
  (set! multihead-attn (attention))
  (set! linear1 (Linear d-model ffn-width #:bias? bias?))
  (set! dropout (Dropout #:p p))
  (set! linear2 (Linear ffn-width d-model #:bias? bias?))
  (set! norm1 (norm))
  (set! norm2 (norm))
  (set! norm3 (norm))
  (set! dropout1 (Dropout #:p p))
  (set! dropout2 (Dropout #:p p))
  (set! dropout3 (Dropout #:p p))
  (set! activation (activation->procedure activation))
  (set! check (checker 'TransformerDecoderLayer
                       (decoder-call/c d-model heads batch-first?)))
  #:forward (tgt memory
                 #:tgt-mask [tgt-mask #f]
                 #:memory-mask [memory-mask #f]
                 #:tgt-key-padding-mask [tgt-padding #f]
                 #:memory-key-padding-mask [memory-padding #f]
                 #:tgt-causal? [tgt-causal? #f]
                 #:memory-causal? [memory-causal? #f])
  (check tgt memory tgt-mask memory-mask tgt-padding memory-padding
         tgt-causal? memory-causal?)
  (~> tgt
      (residual norm-first? norm1
                (lambda (x)
                  (dropout1 (self-attn x x x
                                       #:attn-mask tgt-mask
                                       #:key-padding-mask tgt-padding
                                       #:causal? tgt-causal?))))
      (residual norm-first? norm2
                (lambda (x)
                  (dropout2 (multihead-attn x memory memory
                                            #:attn-mask memory-mask
                                            #:key-padding-mask memory-padding
                                            #:causal? memory-causal?))))
      (residual norm-first? norm3
                (lambda (x)
                  (~> x linear1 activation dropout linear2 dropout3)))))

(define (copy-of prototype make-layer)
  (define copy (call-without-drawing make-layer))
  (with-no-grad
    (for ([to (in-list (parameters copy))]
          [from (in-list (parameters prototype))])
      (copy! to from)))
  copy)

(define (stacked make-layer n copies?)
  (define first-layer (make-layer))
  (cons first-layer
        (for/list ([_ (in-range (sub1 n))])
          (if copies? (copy-of first-layer make-layer) (make-layer)))))

(define-layer TransformerEncoder (layers norm) ;; noqa
  #:contract (stack/c transformer-encoder-layer? transformer-encoder?)
  #:init (make-layer #:layers n #:norm [norm #f] #:copies? [copies? #t])
  (set! layers (LayerList (stacked make-layer n copies?)))
  #:forward (src
             #:mask [mask #f]
             #:key-padding-mask [padding #f]
             #:causal? [causal? #f])
  (normed norm
          (for/fold ([x src]) ([layer (in-layers layers)])
            (layer x #:mask mask #:key-padding-mask padding
                   #:causal? causal?))))

(define-layer TransformerDecoder (layers norm) ;; noqa
  #:contract (stack/c transformer-decoder-layer? transformer-decoder?)
  #:init (make-layer #:layers n #:norm [norm #f] #:copies? [copies? #t])
  (set! layers (LayerList (stacked make-layer n copies?)))
  #:forward (tgt memory
                 #:tgt-mask [tgt-mask #f]
                 #:memory-mask [memory-mask #f]
                 #:tgt-key-padding-mask [tgt-padding #f]
                 #:memory-key-padding-mask [memory-padding #f]
                 #:tgt-causal? [tgt-causal? #f]
                 #:memory-causal? [memory-causal? #f])
  (normed norm
          (for/fold ([x tgt]) ([layer (in-layers layers)])
            (layer x memory
                   #:tgt-mask tgt-mask
                   #:memory-mask memory-mask
                   #:tgt-key-padding-mask tgt-padding
                   #:memory-key-padding-mask memory-padding
                   #:tgt-causal? tgt-causal?
                   #:memory-causal? memory-causal?))))
