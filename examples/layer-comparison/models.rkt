#lang racket/base

(require (only-in racket/contract/base -> ->* ->i </c >=/c and/c)
         (only-in racket/match match-define)
         (only-in torch
                  + * / @ T eq gelu masked-fill narrow ones permute relu reshape
                  softmax tensor-shape transpose tril ~>)
         (only-in torch/generated mean-dim)
         (only-in torch/nn
                  Buffer Conv2d Dropout LayerList LayerNorm Parameter
                  define-layer in-layers kaiming-uniform uniform-init))

(define dropout/c (and/c real? (>=/c 0) (</c 1)))

(define-layer Projection (weight bias)
  #:contract (-> exact-positive-integer? exact-positive-integer? projection?)
  #:init (in-features out-features)
  (set! weight (Parameter (kaiming-uniform (list out-features in-features))))
  (define bound (/ 1.0 (sqrt in-features)))
  (set! bias (Parameter (uniform-init (list out-features) (- bound) bound)))
  #:forward (x)
  (~> x (@ (T weight)) (+ bias)))

(define-layer ChannelNorm (norm)
  #:contract (-> exact-positive-integer? channel-norm?)
  #:init (channels)
  (set! norm (LayerNorm channels))
  #:forward (x)
  (~> x (permute 0 2 3 1) norm (permute 0 3 1 2)))

(define-layer ResidualBlock (conv1 norm1 conv2 norm2 shortcut)
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:stride exact-positive-integer?] residual-block?)
  #:init (in-channels out-channels #:stride [stride 1])
  (set! conv1 (Conv2d in-channels out-channels 3 #:stride stride #:padding 1))
  (set! norm1 (ChannelNorm out-channels))
  (set! conv2 (Conv2d out-channels out-channels 3 #:padding 1))
  (set! norm2 (ChannelNorm out-channels))
  (set! shortcut
        (and (not (and (= stride 1) (= in-channels out-channels)))
             (Conv2d in-channels out-channels 1 #:stride stride)))
  #:forward (x)
  (define residual (if shortcut (shortcut x) x))
  (~> x conv1 norm1 relu conv2 norm2 (+ residual) relu))

(define-layer ResidualStage (blocks)
  #:contract (->* [exact-positive-integer? exact-positive-integer?
                   exact-positive-integer?]
                  [#:stride exact-positive-integer?] residual-stage?)
  #:init (in-channels out-channels depth #:stride [stride 1])
  (set! blocks
        (LayerList
         (for/list ([i (in-range depth)])
           (ResidualBlock (if (zero? i) in-channels out-channels)
                          out-channels #:stride (if (zero? i) stride 1)))))
  #:forward (x)
  (for/fold ([h x]) ([block (in-layers blocks)])
    (block h)))

(define-layer SmallResNet (stem stage1 stage2 stage3 head)
  #:contract (->* [] [#:classes exact-positive-integer?] small-res-net?)
  #:init (#:classes [classes 10])
  (set! stem (Conv2d 3 16 3 #:padding 1))
  (set! stage1 (ResidualStage 16 16 2))
  (set! stage2 (ResidualStage 16 32 2 #:stride 2))
  (set! stage3 (ResidualStage 32 64 2 #:stride 2))
  (set! head (Projection 64 classes))
  #:forward (images)
  (~> images stem relu stage1 stage2 stage3 (mean-dim '(2 3) #f #f) head))

(define-layer CausalSelfAttention
  (width heads head-dim q k v out mask attention-drop output-drop)
  #:contract
  (->i ([width exact-positive-integer?]
        [heads (width) (and/c exact-positive-integer?
                             (lambda (h) (zero? (remainder width h))))]
        [max-t exact-positive-integer?])
       (#:dropout [dropout dropout/c])
       [result causal-self-attention?])
  #:init (width heads max-t #:dropout [dropout 0.1])
  (set! head-dim (quotient width heads))
  (set! q (Projection width width))
  (set! k (Projection width width))
  (set! v (Projection width width))
  (set! out (Projection width width))
  (set! mask (Buffer (eq (tril (ones max-t max-t)) 0)))
  (set! attention-drop (Dropout #:p dropout))
  (set! output-drop (Dropout #:p dropout))
  #:forward (x)
  (match-define (list batch time _) (tensor-shape x))
  (define (split-heads projection)
    (~> x projection (reshape batch time heads head-dim) (transpose 1 2)))
  (define queries (split-heads q))
  (define keys (split-heads k))
  (define values (split-heads v))
  (define scores (/ (@ queries (transpose keys 2 3)) (sqrt head-dim)))
  (define active-mask (narrow (narrow mask 0 0 time) 1 0 time))
  (define weights
    (~> scores (masked-fill active-mask -inf.0) (softmax -1) attention-drop))
  (~> (@ weights values) (transpose 1 2) (reshape batch time width) out output-drop))

(define-layer FeedForward (up down drop)
  #:contract (->* [exact-positive-integer?] [#:dropout dropout/c] feed-forward?)
  #:init (width #:dropout [dropout 0.1])
  (set! up (Projection width (* 4 width)))
  (set! down (Projection (* 4 width) width))
  (set! drop (Dropout #:p dropout))
  #:forward (x)
  (~> x up gelu down drop))

(define-layer TransformerBlock (norm1 attention norm2 mlp)
  #:contract (->* [exact-positive-integer? exact-positive-integer?
                   exact-positive-integer?]
                  [#:dropout dropout/c] transformer-block?)
  #:init (width heads max-t #:dropout [dropout 0.1])
  (set! norm1 (LayerNorm width))
  (set! attention (CausalSelfAttention width heads max-t #:dropout dropout))
  (set! norm2 (LayerNorm width))
  (set! mlp (FeedForward width #:dropout dropout))
  #:forward (x)
  (define h (~> x norm1 attention (+ x)))
  (~> h norm2 mlp (+ h)))

(define-layer TransformerStack (blocks norm)
  #:contract (->* [exact-positive-integer? exact-positive-integer?
                   exact-positive-integer? exact-positive-integer?]
                  [#:dropout dropout/c] transformer-stack?)
  #:init (width heads depth max-t #:dropout [dropout 0.1])
  (set! blocks
        (LayerList
         (for/list ([_ (in-range depth)])
           (TransformerBlock width heads max-t #:dropout dropout))))
  (set! norm (LayerNorm width))
  #:forward (tokens)
  (define h
    (for/fold ([h tokens]) ([block (in-layers blocks)])
      (block h)))
  (~> h norm))
