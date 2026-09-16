#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i </c >=/c and/c any/c between/c contract-out
                  flat-named-contract listof or/c unsupplied-arg?)
         (only-in racket/math infinite? nan? pi)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  add arange cat cos dtype exp index-select length log matmul
                  mul reshape shape silu sin softmax sqrt sub tensor
                  tensor-device tensor? to-dtype transpose unsqueeze
                  upsample-nearest2d)
         (only-in "../nn.rkt"
                  Conv2d Dropout Embedding GroupNorm LayerList Linear
                  define-layer in-layers parameters)
         (only-in "../private/contract.rkt" define/contract-out))

(struct schedule (steps betas alphas alpha-bars) ;; noqa
  #:constructor-name make-schedule
  #:omit-define-syntaxes)

(provide (contract-out
          [schedule? (-> any/c boolean?)]
          [schedule-steps (-> schedule? exact-positive-integer?)]
          [schedule-betas (-> schedule? tensor?)]
          [schedule-alphas (-> schedule? tensor?)]
          [schedule-alpha-bars (-> schedule? tensor?)]))

(define (betas->schedule betas)
  (define alphas (for/list ([b (in-list betas)]) (- 1.0 b)))
  (define alpha-bars
    (let loop ([as alphas] [acc 1.0] [out '()])
      (cond
        [(null? as) (reverse out)]
        [else
         (define next (* acc (car as)))
         (loop (cdr as) next (cons next out))])))
  (make-schedule (length betas) (tensor betas) (tensor alphas) (tensor alpha-bars)))

(define variance/c
  (flat-named-contract 'variance (and/c real? (between/c 0.0 1.0))))

(define offset/c
  (flat-named-contract
   'finite-nonnegative-real
   (lambda (v)
     (and (real? v) (>= v 0)
          (let ([f (exact->inexact v)]) (not (or (nan? f) (infinite? f))))))))

(define timesteps/c
  (flat-named-contract
   'int64-vector
   (lambda (v)
     (and (tensor? v) (eq? (dtype v) 'int64) (= 1 (length (shape v)))))))

(define/contract-out (linear-schedule [steps 1000] ;; noqa
                                      #:beta-start [beta-start 1e-4]
                                      #:beta-end [beta-end 0.02])
  (->* []
       [exact-positive-integer? #:beta-start variance/c #:beta-end variance/c]
       schedule?)
  (define span (max 1 (sub1 steps)))
  (betas->schedule
   (for/list ([i (in-range steps)])
     (exact->inexact (+ beta-start (* (- beta-end beta-start) (/ i span)))))))

(define/contract-out (cosine-schedule [steps 1000] #:offset [offset 0.008]) ;; noqa
  (->* [] [exact-positive-integer? #:offset offset/c] schedule?)
  (define (f t)
    (define x (* (/ (+ (/ t steps) offset) (+ 1.0 offset)) (/ pi 2.0)))
    (* (cos x) (cos x)))
  (betas->schedule
   (for/list ([t (in-range steps)])
     (min 0.999 (- 1.0 (/ (f (add1 t)) (f t)))))))

(define/contract-out (q-sample sched x0 t noise) ;; noqa
  (-> schedule? tensor? timesteps/c tensor? tensor?)
  (define a (reshape (index-select (schedule-alpha-bars sched) 0 t) -1 1 1 1))
  (add (mul (sqrt a) x0) (mul (sqrt (sub 1.0 a)) noise)))

(define even-dim/c
  (flat-named-contract 'even-positive-integer
                       (lambda (n) (and (exact-positive-integer? n) (even? n)))))

(define/contract-out (sinusoidal-embedding t dim) ;; noqa
  (-> timesteps/c even-dim/c tensor?)
  (define half (quotient dim 2))
  (define freqs
    (exp (mul (arange half #:device (tensor-device t))
              (- (/ (log 10000.0) half)))))
  (define angles (mul (unsqueeze (to-dtype t 'float32) 1) (unsqueeze freqs 0)))
  (cat (list (sin angles) (cos angles)) 1))

(define-layer TimeEmbedding (dim fc1 fc2) ;; noqa
  #:contract (-> even-dim/c time-embedding?)
  #:init (dim)
  (set! fc1 (Linear dim (* 4 dim)))
  (set! fc2 (Linear (* 4 dim) (* 4 dim)))
  #:forward (t)
  (~> t
      (sinusoidal-embedding dim)
      (to-dtype (dtype (car (parameters fc1))))
      fc1
      silu
      fc2))

(define norm-groups 32)

(define channels/c
  (flat-named-contract 'multiple-of-32
                       (lambda (n) (and (exact-positive-integer? n)
                                        (zero? (remainder n norm-groups))))))

(define dropout/c (and/c real? (>=/c 0) (</c 1)))

(define levels/c
  (flat-named-contract 'at-most-five-levels-of-32x32
                       (lambda (l) (and (list? l) (<= 1 (length l) 5)
                                        (andmap exact-positive-integer? l)))))

(define (resolution i) (quotient 32 (expt 2 i)))

(define (reachable? attention mults)
  (for/and ([r (in-list attention)])
    (for/or ([i (in-range (length mults))])
      (= r (resolution i)))))

(define-layer ResBlock (norm1 conv1 emb norm2 drop conv2 skip) ;; noqa
  #:contract (->* [channels/c channels/c exact-positive-integer?]
                  [#:dropout dropout/c]
                  res-block?)
  #:init (in out t-dim #:dropout [dropout 0.0])
  (set! norm1 (GroupNorm norm-groups in))
  (set! conv1 (Conv2d in out 3 #:padding 1))
  (set! emb (Linear t-dim out))
  (set! norm2 (GroupNorm norm-groups out))
  (set! drop (Dropout #:p dropout))
  (set! conv2 (Conv2d out out 3 #:padding 1))
  (set! skip (and (not (= in out)) (Conv2d in out 1)))
  #:forward (x temb)
  (define h (~> x norm1 silu conv1))
  (define shift (~> temb silu emb (reshape (length temb) -1 1 1)))
  (add (~> (add h shift) norm2 silu drop conv2) (if skip (skip x) x)))

(define-layer AttentionBlock (norm q k v proj) ;; noqa
  #:contract (-> channels/c attention-block?)
  #:init (channels)
  (set! norm (GroupNorm norm-groups channels))
  (set! q (Conv2d channels channels 1))
  (set! k (Conv2d channels channels 1))
  (set! v (Conv2d channels channels 1))
  (set! proj (Conv2d channels channels 1))
  #:forward (x)
  (define dims (shape x))
  (define n (car dims))
  (define c (cadr dims))
  (define pixels (* (caddr dims) (cadddr dims)))
  (define normed (norm x))
  (define (tokens layer) (reshape (layer normed) n c pixels))
  (define scores (mul (matmul (transpose (tokens q) 1 2) (tokens k))
                      (/ 1.0 (sqrt c))))
  (define weights (transpose (softmax scores -1) 1 2))
  (define out (reshape (matmul (tokens v) weights) n c (caddr dims) (cadddr dims)))
  (add x (proj out)))

(define-layer Downsample (conv) ;; noqa
  #:contract (-> channels/c downsample?)
  #:init (channels)
  (set! conv (Conv2d channels channels 3 #:stride 2 #:padding 1))
  #:forward (x _temb)
  (conv x))

(define-layer Upsample (conv) ;; noqa
  #:contract (-> channels/c upsample?)
  #:init (channels)
  (set! conv (Conv2d channels channels 3 #:padding 1))
  #:forward (x)
  (conv (upsample-nearest2d x)))

(define-layer Stage (res attn)
  #:init (res attn)
  #:forward (x temb)
  (define res-out (res x temb))
  (if attn (attn res-out) res-out))

(define-layer UNet (time classes in-conv downs mid1 mid-attn mid2 ups ;; noqa
                    out-norm out-conv)
  #:contract (->i ()
                  (#:base [base channels/c]
                   #:mults [mults levels/c]
                   #:blocks [blocks exact-positive-integer?]
                   #:attention [attention (listof exact-positive-integer?)]
                   #:dropout [dropout dropout/c]
                   #:classes [classes (or/c #f exact-positive-integer?)])
                  #:pre/name (mults attention)
                  "every attention resolution must be one of the levels' resolutions"
                  (reachable? (if (unsupplied-arg? attention) '(16) attention)
                              (if (unsupplied-arg? mults) '(1 2 2 2) mults))
                  [result unet?])
  #:init (#:base [base 128] #:mults [mults '(1 2 2 2)] #:blocks [blocks 2]
          #:attention [attention '(16)] #:dropout [dropout 0.1]
          #:classes [n-classes #f])
  (define t-dim (* 4 base))
  (define levels (length mults))
  (define (width i) (* base (list-ref mults i)))
  (define (stage narrow wide res)
    (Stage (ResBlock narrow wide t-dim #:dropout dropout)
           (and (memv res attention) (AttentionBlock wide))))
  (set! time (TimeEmbedding base))
  (set! classes (and n-classes (Embedding (add1 n-classes) t-dim)))
  (set! in-conv (Conv2d 3 base 3 #:padding 1))
  (define stages '())
  (define skips (list base))
  (define channels base)
  (for ([i (in-range levels)])
    (define wide (width i))
    (for ([_ (in-range blocks)])
      (set! stages (cons (stage channels wide (resolution i)) stages))
      (set! skips (cons wide skips))
      (set! channels wide))
    (unless (= i (sub1 levels))
      (set! stages (cons (Downsample wide) stages))
      (set! skips (cons wide skips))))
  (set! downs (LayerList (reverse stages)))
  (set! mid1 (ResBlock channels channels t-dim #:dropout dropout))
  (set! mid-attn (AttentionBlock channels))
  (set! mid2 (ResBlock channels channels t-dim #:dropout dropout))
  (set! stages '())
  (for ([i (in-range (sub1 levels) -1 -1)])
    (define wide (width i))
    (for ([_ (in-range (add1 blocks))])
      (set! stages (cons (stage (+ channels (car skips)) wide (resolution i)) stages))
      (set! skips (cdr skips))
      (set! channels wide))
    (unless (zero? i)
      (set! stages (cons (Upsample wide) stages))))
  (set! ups (LayerList (reverse stages)))
  (set! out-norm (GroupNorm norm-groups (width 0)))
  (set! out-conv (Conv2d (width 0) 3 3 #:padding 1))
  #:forward (x t y)
  (define temb
    (let ([te (time t)])
      (if classes (add te (classes y)) te)))
  (define x0 (in-conv x))
  (define-values (bottom stack)
    (for/fold ([down x0] [stack (list x0)]) ([layer (in-layers downs)])
      (define next (layer down temb))
      (values next (cons next stack))))
  (define middle (mid2 (mid-attn (mid1 bottom temb)) temb))
  (define-values (top _rest)
    (for/fold ([up middle] [rest stack]) ([layer (in-layers ups)])
      (if (upsample? layer)
          (values (layer up) rest)
          (values (layer (cat (list up (car rest)) 1) temb) (cdr rest)))))
  (~> top out-norm silu out-conv))
