#lang racket/base

(require (only-in racket/contract/base
                  -> ->i </c >=/c and/c any contract flat-contract-predicate
                  flat-named-contract list/c or/c)
         (only-in racket/list append-map drop-right last)
         (only-in racket/match match match-define)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  + cat copy! dtype flatten narrow permute reshape select shape
                  transpose upsample-bicubic2d with-no-grad zero!)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../foreign/ops.rkt" any-float-dtype/c)
         (only-in "../generated.rkt" broadcast-to)
         (only-in "../nn/conv.rkt" Conv2d)
         (only-in "../nn/init.rkt" trunc-normal-init)
         (only-in "../nn/layer.rkt"
                  children define-layer named-parameters)
         (only-in "../nn/layer-list.rkt" LayerList)
         (only-in "../nn/layer-norm.rkt" LayerNorm)
         (only-in "../nn/linear.rkt" Linear linear?)
         (only-in "../nn/parameter.rkt" Parameter)
         (only-in "../nn/transformer.rkt" TransformerEncoderLayer)
         (only-in "../private/contract.rkt" define/contract-out supplied-or))

(define image-size/c
  (flat-named-contract
   'image-size
   (or/c exact-positive-integer?
         (list/c exact-positive-integer? exact-positive-integer?))))

(define output/c (or/c 'head 'cls 'tokens 'attention))

(define float-dtype? (flat-contract-predicate any-float-dtype/c))

(define (sides image-size)
  (if (list? image-size) image-size (list image-size image-size)))

(define (image-fits x in-channels patch)
  (match-define (list _ channels h w) (shape x))
  (cond
    [(not (float-dtype? (dtype x)))
     (format "x is a floating-point batch, not ~a" (dtype x))]
    [(not (= channels in-channels))
     (format "x has ~a channels where the model reads ~a" channels in-channels)]
    [(not (and (positive? h) (positive? w)
               (zero? (remainder h patch)) (zero? (remainder w patch))))
     (format "each side of x is a positive multiple of the patch ~a, not ~a by ~a"
             patch h w)]
    [else #t]))

(define (patch-call/c in-channels patch)
  (->i ([x image-batch/c])
       #:pre/desc (x) (image-fits x in-channels patch)
       any))

(define (vit-call/c in-channels patch)
  (->i ([x image-batch/c] [output output/c])
       #:pre/desc (x) (image-fits x in-channels patch)
       any))

(module+ private
  (provide positions)) ;; noqa

(define (positions pos-embed grid target)
  (match-define (list gh gw) grid)
  (define width (last (shape pos-embed)))
  (cond
    [(equal? grid target) pos-embed]
    [else
     (cat (list (narrow pos-embed 1 0 1)
                (~> (narrow pos-embed 1 1 (* gh gw))
                    (reshape 1 gh gw width)
                    (permute 0 3 1 2)
                    (upsample-bicubic2d target)
                    (permute 0 2 3 1)
                    (reshape 1 -1 width)))
          1)]))

(define (checker who call/c)
  (contract call/c void who 'caller who #f))

(define (truncated-normal dims)
  (Parameter (trunc-normal-init dims #:std 0.02)))

(define (linears m)
  (if (linear? m) (list m) (append-map linears (children m))))

(define (reinitialise! linear)
  (match-define (list (cons _ weight) (cons _ bias)) (named-parameters linear))
  (with-no-grad
    (copy! weight (trunc-normal-init (shape weight) #:std 0.02))
    (zero! bias)))

(define-layer PatchEmbed (proj patch check) ;; noqa
  #:predicate patch-embed?
  #:contract (-> exact-positive-integer? exact-positive-integer?
                 exact-positive-integer? patch-embed?)
  #:init (in-channels width patch-side)
  (set! proj (Conv2d in-channels width patch-side #:stride patch-side))
  (set! patch patch-side)
  (set! check (checker 'PatchEmbed (patch-call/c in-channels patch-side)))
  #:forward (x)
  (check x)
  (~> x proj (flatten 2) (transpose 1 2)))

(define (encoded z layers)
  (for/fold ([z z]) ([layer (in-list layers)])
    (layer z)))

(define (last-attention z layers)
  (define-values (_ weights)
    ((last layers) (encoded z (drop-right layers 1))
                   #:need-weights? #t #:average-attn-weights? #f))
  weights)

(define (read-out output normed head)
  (define cls (select normed 1 0))
  (match output
    ['tokens normed]
    ['cls cls]
    ['head (if head (head cls) cls)]))

(define-layer VisionTransformer ;; noqa
  (patch-embed cls-token pos-embed blocks norm head patch grid check)
  #:predicate vision-transformer?
  #:contract
  (->i ()
       (#:image-size [image-size image-size/c]
        #:patch [patch exact-positive-integer?]
        #:in-channels [in-channels exact-positive-integer?]
        #:width [width exact-positive-integer?]
        #:depth [depth exact-positive-integer?]
        #:heads [heads exact-positive-integer?]
        #:ffn-width [ffn-width exact-positive-integer?]
        #:dropout [dropout (and/c real? (>=/c 0) (</c 1))]
        #:layer-norm-eps [eps (and/c real? positive?)]
        #:classes [classes (or/c #f exact-positive-integer?)])
       #:pre/name (image-size patch) "#:patch divides each side of #:image-size"
       (let ([p (supplied-or patch 16)])
         (for/and ([side (in-list (sides (supplied-or image-size 224)))])
           (zero? (remainder side p))))
       #:pre/name (width heads) "#:heads divides #:width"
       (zero? (remainder (supplied-or width 768) (supplied-or heads 12)))
       [_ vision-transformer?])
  #:init (#:image-size [image-size 224] #:patch [patch-side 16]
          #:in-channels [in-channels 3]
          #:width [width 768] #:depth [depth 12] #:heads [heads 12]
          #:ffn-width [ffn-width (* 4 width)] #:dropout [dropout 0.0]
          #:layer-norm-eps [eps 1e-6] #:classes [classes #f])
  (set! patch patch-side)
  (set! grid (map (lambda (side) (quotient side patch-side)) (sides image-size)))
  (set! patch-embed (PatchEmbed in-channels width patch-side))
  (set! blocks
        (LayerList
         (for/list ([_ (in-range depth)])
           (TransformerEncoderLayer width #:heads heads
                                    #:ffn-width ffn-width
                                    #:dropout dropout
                                    #:activation 'gelu
                                    #:norm-first? #t
                                    #:layer-norm-eps eps
                                    #:batch-first? #t))))
  (set! norm (LayerNorm width #:eps eps))
  (set! head (and classes (Linear width classes)))
  (set! cls-token (truncated-normal (list 1 1 width)))
  (set! pos-embed (truncated-normal (list 1 (+ 1 (apply * grid)) width)))
  (for-each reinitialise! (linears blocks))
  (when head (reinitialise! head))
  (set! check (checker 'VisionTransformer (vit-call/c in-channels patch-side)))
  #:forward (x #:output [output 'head])
  (check x output)
  (match-define (list n _ h w) (shape x))
  (define tokens (patch-embed x))
  (define z
    (+ (cat (list (broadcast-to cls-token (list n 1 (last (shape tokens))))
                  tokens)
            1)
       (positions pos-embed grid (list (quotient h patch) (quotient w patch)))))
  (define layers (children blocks))
  (if (eq? output 'attention)
      (last-attention z layers)
      (read-out output (norm (encoded z layers)) head)))

(define huggingface-renames
  '((#rx"^embeddings[.]cls_token$" . "cls-token")
    (#rx"^embeddings[.]position_embeddings$" . "pos-embed")
    (#rx"^embeddings[.]patch_embeddings[.]projection[.]" . "patch-embed.proj.")
    (#rx"^encoder[.]layer[.]([0-9]+)[.]layernorm_before[.]"
     . "blocks.\\1.norm1.")
    (#rx"^encoder[.]layer[.]([0-9]+)[.]attention[.]attention[.](query|key|value)[.]"
     . "blocks.\\1.self-attn.\\2.")
    (#rx"^encoder[.]layer[.]([0-9]+)[.]attention[.]output[.]dense[.]"
     . "blocks.\\1.self-attn.out.")
    (#rx"^encoder[.]layer[.]([0-9]+)[.]layernorm_after[.]"
     . "blocks.\\1.norm2.")
    (#rx"^encoder[.]layer[.]([0-9]+)[.]intermediate[.]dense[.]"
     . "blocks.\\1.linear1.")
    (#rx"^encoder[.]layer[.]([0-9]+)[.]output[.]dense[.]"
     . "blocks.\\1.linear2.")
    (#rx"^layernorm[.]" . "norm.")
    (#rx"^classifier[.]" . "head.")))

(define/contract-out (huggingface-vit-key key) ;; noqa
  (-> string? (or/c string? #f))
  (define bare (regexp-replace #rx"^vit[.]" key ""))
  (define rename
    (findf (lambda (r) (regexp-match? (car r) bare)) huggingface-renames))
  (cond
    [(regexp-match? #rx"^pooler[.]" bare) #f]
    [rename (regexp-replace (car rename) bare (cdr rename))]
    [else key]))
