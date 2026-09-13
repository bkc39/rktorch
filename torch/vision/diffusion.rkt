#lang racket/base

(require (only-in racket/contract/base
                  -> ->* and/c any/c between/c contract-out
                  flat-named-contract listof)
         (only-in racket/math infinite? nan? pi)
         (only-in "../foreign.rkt"
                  add arange cat cos dtype exp index-select length log mul
                  reshape shape silu sin sqrt sub tensor tensor-device tensor?
                  to-dtype unsqueeze)
         (only-in "../nn.rkt"
                  Conv2d ConvTranspose2d GroupNorm Linear define-layer
                  parameters)
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

;; the closed form q(x_t | x_0) needs the cumulative products as tensors on
;; the default device, so a schedule is built where its model lives
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
   (lambda (v) (and (real? v) (not (nan? v)) (not (infinite? v)) (>= v 0)))))

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
  (define features
    (to-dtype (sinusoidal-embedding t dim) (dtype (car (parameters fc1)))))
  (fc2 (silu (fc1 features))))

(define channels/c
  (flat-named-contract 'multiple-of-eight
                       (lambda (n) (and (exact-positive-integer? n)
                                        (zero? (remainder n 8))))))

(define-layer ResBlock (norm1 conv1 emb norm2 conv2 skip) ;; noqa
  #:contract (-> channels/c channels/c exact-positive-integer? res-block?)
  #:init (in out t-dim)
  (set! norm1 (GroupNorm 8 in))
  (set! conv1 (Conv2d in out 3 #:padding 1))
  (set! emb (Linear t-dim out))
  (set! norm2 (GroupNorm 8 out))
  (set! conv2 (Conv2d out out 3 #:padding 1))
  (set! skip (and (not (= in out)) (Conv2d in out 1)))
  #:forward (x temb)
  (define h (conv1 (silu (norm1 x))))
  (define shifted (add h (reshape (emb (silu temb)) (length temb) -1 1 1)))
  (add (conv2 (silu (norm2 shifted))) (if skip (skip x) x)))

(define-layer UNet (time in-conv down1 pool1 down2 pool2 mid ;; noqa
                    up2-conv up2 up1-conv up1 out-norm out-conv)
  #:contract (->* [] [#:base channels/c] unet?)
  #:init (#:base [base 32])
  (define t-dim (* 4 base))
  (set! time (TimeEmbedding base))
  (set! in-conv (Conv2d 3 base 3 #:padding 1))
  (set! down1 (ResBlock base base t-dim))
  (set! pool1 (Conv2d base base 3 #:stride 2 #:padding 1))
  (set! down2 (ResBlock base (* 2 base) t-dim))
  (set! pool2 (Conv2d (* 2 base) (* 2 base) 3 #:stride 2 #:padding 1))
  (set! mid (ResBlock (* 2 base) (* 2 base) t-dim))
  (set! up2-conv (ConvTranspose2d (* 2 base) (* 2 base) 4 #:stride 2 #:padding 1))
  (set! up2 (ResBlock (* 4 base) (* 2 base) t-dim))
  (set! up1-conv (ConvTranspose2d (* 2 base) base 4 #:stride 2 #:padding 1))
  (set! up1 (ResBlock (* 2 base) base t-dim))
  (set! out-norm (GroupNorm 8 base))
  (set! out-conv (Conv2d base 3 3 #:padding 1))
  #:forward (x t)
  (define temb (time t))
  (define h1 (down1 (in-conv x) temb))
  (define h2 (down2 (pool1 h1) temb))
  (define h3 (mid (pool2 h2) temb))
  (define u2 (up2 (cat (list (up2-conv h3) h2) 1) temb))
  (define u1 (up1 (cat (list (up1-conv u2) h1) 1) temb))
  (out-conv (silu (out-norm u1))))
