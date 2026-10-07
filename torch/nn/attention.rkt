#lang racket/base

(require (only-in racket/contract/base
                  ->i </c >=/c and/c any contract flat-named-contract or/c)
         (only-in racket/list last)
         (only-in racket/match match match-define)
         (only-in "../foreign.rkt"
                  add copy! device dtype masked-fill matmul mul narrow ones
                  permute reshape scaled-dot-product-attention shape softmax
                  squeeze tensor? to-dtype transpose triu unsqueeze
                  with-no-grad zero! zeros zeros-like)
         (only-in "../generated.rkt" dropout mean-dim)
         (only-in "init.rkt" uniform-init)
         (only-in "layer.rkt" define-layer named-parameters training? with-mode)
         (only-in "linear.rkt" Linear)
         (only-in (submod "linear.rkt" private) tensors->Linear))

(define (xavier-uniform dims)
  (match-define (list fan-out fan-in) dims)
  (define std (sqrt (/ 2.0 (exact->inexact (+ fan-in fan-out)))))
  (define bound (* (sqrt 3.0) std))
  (uniform-init dims (- bound) bound))

(define (rows packed start count)
  (define t (zeros count (cadr (shape packed))))
  (copy! t (narrow packed 0 start count))
  t)

;; nn.MultiheadAttention._reset_parameters: one xavier draw over the packed
;; [3E, E] in_proj_weight when the key and value are E wide, else one per
;; projection, query first.
(define (draw-in-projection embed-dim key-dim value-dim)
  (cond
    [(= embed-dim key-dim value-dim)
     (define packed (xavier-uniform (list (* 3 embed-dim) embed-dim)))
     (for/list ([i (in-range 3)])
       (rows packed (* i embed-dim) embed-dim))]
    [else
     (for/list ([width (in-list (list embed-dim key-dim value-dim))])
       (xavier-uniform (list embed-dim width)))]))

(define (zero-bias! layer)
  (with-no-grad
    (zero! (cdr (assoc "bias" (named-parameters layer))))))

(define attention-sequence/c
  (flat-named-contract
   'attention-sequence
   (lambda (x) (and (tensor? x) (memv (length (shape x)) '(2 3)) #t))))

(define attention-mask/c
  (flat-named-contract
   'attention-mask
   (lambda (m)
     (and (tensor? m)
          (memq (dtype m) '(bool float32 float64 float16 bfloat16))
          #t))))

(define (rank x)
  (length (shape x)))

(define (batch+length x batch-first?)
  (match (shape x)
    [(list l _) (list #f l)]
    [(list a b _) (if batch-first? (list a b) (list b a))]))

(define (wide who x expected)
  (define width (last (shape x)))
  (or (= width expected)
      (format "~a is ~a wide, not ~a" who width expected)))

(define (shaped who m expected)
  (or (not m)
      (and (member (shape m) expected) #t)
      (format "~a has shape ~a, not ~a" who (shape m)
              (if (null? (cdr expected))
                  (car expected)
                  (format "~a or ~a" (car expected) (cadr expected))))))

(define (same-batch who-q q who-k k batch-first?)
  (define n (car (batch+length q batch-first?)))
  (define m (car (batch+length k batch-first?)))
  (or (not (and n m))
      (= n m)
      (format "~a has batch ~a and ~a ~a" who-q n who-k m)))

(define (padding-shaped who mask keys batch-first?)
  (match-define (list n s) (batch+length keys batch-first?))
  (shaped who mask (list (if n (list n s) (list s)))))

(define (attn-mask-shaped who mask queries keys heads batch-first?)
  (match-define (list n l) (batch+length queries batch-first?))
  (match-define (list _ s) (batch+length keys batch-first?))
  (shaped who mask (list (list l s) (list (* (or n 1) heads) l s))))

(define (call/c embed-dim key-dim value-dim heads batch-first?)
  (->i ([query attention-sequence/c]
        [key attention-sequence/c]
        [value attention-sequence/c]
        [key-padding-mask (or/c #f attention-mask/c)]
        [attn-mask (or/c #f attention-mask/c)]
        [causal? boolean?]
        [need-weights? boolean?]
        [average-attn-weights? boolean?])
       #:pre/desc (query) (wide "query" query embed-dim)
       #:pre/desc (key) (wide "key" key key-dim)
       #:pre/desc (value) (wide "value" value value-dim)
       #:pre/desc (query key value)
       (or (= (rank query) (rank key) (rank value))
           (string-append "query, key and value are all batched (rank 3)"
                          " or all unbatched (rank 2)"))
       #:pre/desc (query key value)
       (let ([k (batch+length key batch-first?)]
             [v (batch+length value batch-first?)])
         (or (not (= (rank key) (rank value)))
             (equal? k v)
             (format "key and value differ in batch or length: ~a and ~a"
                     k v)))
       #:pre/desc (query key value)
       (same-batch "query" query "key" key batch-first?)
       #:pre/desc (query key value key-padding-mask)
       (padding-shaped "key-padding-mask" key-padding-mask key batch-first?)
       #:pre/desc (query key value attn-mask)
       (attn-mask-shaped "attn-mask" attn-mask query key heads batch-first?)
       any))

(module+ private
  (provide attention-mask/c
           attention-sequence/c
           attn-mask-shaped
           padding-shaped
           rank
           same-batch
           wide))

(define (split-heads x heads batch-first?)
  (match-define (list a b width) (shape x))
  (define split (reshape x a b heads (quotient width heads)))
  (if batch-first? (permute split 0 2 1 3) (permute split 1 2 0 3)))

(define (merge-heads x batch-first?)
  (match-define (list n heads l d) (shape x))
  (if batch-first?
      (reshape (permute x 0 2 1 3) n l (* heads d))
      (reshape (permute x 2 0 1 3) l n (* heads d))))

(define (additive mask float-type)
  (if (eq? (dtype mask) 'bool)
      (masked-fill (zeros-like mask #:dtype float-type) mask -inf.0)
      (to-dtype mask float-type)))

(define (causal-mask l s like)
  (define dev (device like))
  (masked-fill (zeros l s #:device dev #:dtype (dtype like))
               (triu (ones l s #:device dev #:dtype 'bool) 1)
               -inf.0))

(define (combined-mask padding mask causal? like heads s)
  (match-define (list n _ l _) (shape like))
  (define float-type (dtype like))
  (define parts
    (append
     (if padding (list (reshape (additive padding float-type) n 1 1 s)) '())
     (if mask
         (list (if (= 3 (rank mask))
                   (reshape (additive mask float-type) n heads l s)
                   (additive mask float-type)))
         '())
     (if causal? (list (causal-mask l s like)) '())))
  (and (pair? parts)
       (for/fold ([sum (car parts)]) ([part (in-list (cdr parts))])
         (add sum part))))

(define (attend-heads qh kh vh mask causal? p need-weights?)
  (cond
    [need-weights?
     (define scale (sqrt (/ 1.0 (last (shape qh)))))
     (define scores (matmul (mul qh scale) (transpose kh 2 3)))
     (define weights (softmax (if mask (add scores mask) scores) 3))
     (define dropped (if (positive? p) (dropout weights p #t) weights))
     (values (matmul dropped vh) dropped)]
    [else
     (values (scaled-dot-product-attention qh kh vh
                                           #:mask mask
                                           #:causal? causal?
                                           #:dropout p)
             #f)]))

(define (attend projections q k v
                #:heads heads
                #:batch-first? batch-first?
                #:dropout p
                #:key-padding-mask padding
                #:attn-mask mask
                #:causal? causal?
                #:need-weights? need-weights?
                #:average? average?)
  (match-define (list query key value out) projections)
  (define unbatched? (= 2 (rank q)))
  (define batch-dim (if batch-first? 0 1))
  (define (heads-of projection x)
    (split-heads (projection (if unbatched? (unsqueeze x batch-dim) x))
                 heads batch-first?))
  (define qh (heads-of query q))
  (define kh (heads-of key k))
  (define vh (heads-of value v))
  (define fused-causal? (and causal? (not (or need-weights? padding mask))))
  (define-values (attended weights)
    (attend-heads qh kh vh
                  (combined-mask padding mask (and causal? (not fused-causal?))
                                 qh heads (caddr (shape kh)))
                  fused-causal? p need-weights?))
  (define projected (out (merge-heads attended batch-first?)))
  (define output (if unbatched? (squeeze projected batch-dim) projected))
  (cond
    [need-weights?
     (define per-query (if average? (mean-dim weights '(1) #f #f) weights))
     (values output (if unbatched? (squeeze per-query 0) per-query))]
    [else output]))

(define-layer MultiheadAttention ;; noqa
  (query key value out heads dropout-p batch-first? check)
  #:contract (->i ([embed-dim exact-positive-integer?]
                   #:heads [heads exact-positive-integer?])
                  (#:dropout [dropout (and/c real? (>=/c 0) (</c 1))]
                   #:bias? [bias? boolean?]
                   #:batch-first? [batch-first? boolean?]
                   #:key-dim [key-dim exact-positive-integer?]
                   #:value-dim [value-dim exact-positive-integer?])
                  #:pre/name (embed-dim heads)
                  "#:heads divides the embedding width"
                  (zero? (remainder embed-dim heads))
                  [_ multihead-attention?])
  #:init (embed-dim
          #:heads heads
          #:dropout [dropout 0.0]
          #:bias? [bias? #t]
          #:batch-first? [batch-first? #f]
          #:key-dim [key-dim embed-dim]
          #:value-dim [value-dim embed-dim])
  (set! out (Linear embed-dim embed-dim #:bias? bias?))
  (when bias? (zero-bias! out))
  (match-define (list wq wk wv)
    (draw-in-projection embed-dim key-dim value-dim))
  (define (projection weight)
    (tensors->Linear weight (and bias? (zeros embed-dim))))
  (set! query (projection wq))
  (set! key (projection wk))
  (set! value (projection wv))
  (set! dropout-p (exact->inexact dropout))
  (set! check
        (contract (call/c embed-dim key-dim value-dim heads batch-first?)
                  void 'MultiheadAttention 'caller 'MultiheadAttention #f))
  #:forward (q k v
             #:key-padding-mask [key-padding-mask #f]
             #:attn-mask [attn-mask #f]
             #:causal? [causal? #f]
             #:need-weights? [need-weights? #f]
             #:average-attn-weights? [average? #t])
  (check q k v key-padding-mask attn-mask causal? need-weights? average?)
  (with-mode
    (attend (list query key value out) q k v
            #:heads heads
            #:batch-first? batch-first?
            #:dropout (if (training? mode) dropout-p 0.0)
            #:key-padding-mask key-padding-mask
            #:attn-mask attn-mask
            #:causal? causal?
            #:need-weights? need-weights?
            #:average? average?)))
