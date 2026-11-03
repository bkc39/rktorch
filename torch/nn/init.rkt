#lang racket/base

(require (only-in racket/contract/base -> ->i and/c listof)
         (only-in racket/match match-define)
         (only-in racket/math infinite? sqr)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * + - / > clamp item log ne randn sum tensor? uniform! where
                  zeros)
         (only-in "../foreign/generator.rkt" generator-on? normal!)
         (only-in "../private/contract.rkt" define/checked-out supplied-or)
         (only-in "../private/erf.rkt" erf))

(provide uniform-bias)

(define dims/c (listof exact-nonnegative-integer?))

;; zeros + uniform! consumes the RNG exactly as torch's empty().uniform_().
(define/checked-out (uniform-init dims low high)
  (->i ([dims dims/c] [low real?] [high real?])
       #:pre/name () "the generator lives on the target device"
       (generator-on? #f #f)
       [result tensor?])
  (define t (apply zeros dims))
  (uniform! t low high)
  t)

;; randn is empty().normal_(): the same RNG consumption as init.normal_.
(define/checked-out (normal-init dims) ;; noqa
  (-> dims/c tensor?)
  (apply randn dims))

(define/checked-out (fan-in dims)
  (-> dims/c exact-nonnegative-integer?)
  (apply * (cdr dims)))

;; the default #:a (sqrt 5) is what nn.Linear.reset_parameters passes
(define/checked-out (kaiming-uniform dims #:a [a (sqrt 5.0)]) ;; noqa
  (->i ([dims dims/c])
       (#:a [a real?])
       #:pre/name () "the generator lives on the target device"
       (generator-on? #f #f)
       [result tensor?])
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-init dims (- bound) bound))

;; the bias of nn.Linear and _ConvNd: U(-1/sqrt fan, 1/sqrt fan)
(define (uniform-bias n fan)
  (define bound (/ 1.0 (sqrt fan)))
  (uniform-init (list n) (- bound) bound))

(define (any? mask)
  (~> mask sum item positive?))

(define (normal-draw dims mean std)
  (define t (apply zeros dims))
  (normal! t mean std)
  t)

(define (redrawn-normal dims mean std a b)
  (let loop ([result (normal-draw dims mean std)])
    (define outside (ne result (clamp result #:min a #:max b)))
    (cond
      [(any? outside) (loop (where outside (normal-draw dims mean std) result))]
      [else result])))

(define (accepted-uniform dims mean std a b)
  (define log-peak (~> (min mean b) (max a) (- mean) (/ std) sqr (* -0.5)))
  (define (rejected candidates)
    (define z (/ (- candidates mean) std))
    (define log-density (- (* z z -0.5) log-peak))
    (~> (uniform-init dims 0.0 1.0) log (> log-density)))
  (define drawn (uniform-init dims a b))
  (let loop ([result drawn] [pending (rejected drawn)])
    (cond
      [(any? pending)
       (define candidates (uniform-init dims a b))
       (loop (where pending candidates result)
             (where pending (rejected candidates) pending))]
      [else result])))

(define (normal-mass mean std a b)
  (define (cdf v)
    (~> (- v mean) (/ std) exact->inexact (/ (sqrt 2.0)) erf (+ 1.0) (/ 2.0)))
  (- (cdf b) (cdf a)))

(define (uniform-branch? mean std a b)
  (~> (normal-mass mean std a b) (> 0.3) not))

(define/checked-out (trunc-normal-init dims ;; noqa
                                       #:mean [mean 0.0]
                                       #:std [std 1.0]
                                       #:a [a -2.0]
                                       #:b [b 2.0])
  (->i ([dims dims/c])
       (#:mean [mean real?]
        #:std [std (and/c real? positive?)]
        #:a [a real?]
        #:b [b real?])
       #:pre/name (a b) "#:a is below #:b"
       (< (supplied-or a -2.0) (supplied-or b 2.0))
       #:pre/name (mean std a b)
       "#:a and #:b are finite when [a, b] holds at most 0.3 of the mass"
       (let ([a (supplied-or a -2.0)] [b (supplied-or b 2.0)])
         (not (and (uniform-branch? (supplied-or mean 0.0) (supplied-or std 1.0)
                                    a b)
                   (or (infinite? a) (infinite? b)))))
       #:pre/name () "the generator lives on the target device"
       (generator-on? #f #f)
       [result tensor?])
  (match-define (list μ σ lo hi) (map exact->inexact (list mean std a b)))
  (cond
    [(uniform-branch? mean std a b) (accepted-uniform dims μ σ lo hi)]
    [else (redrawn-normal dims μ σ lo hi)]))
