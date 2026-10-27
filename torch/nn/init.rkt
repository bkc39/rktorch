#lang racket/base

(require (only-in racket/contract/base -> ->* ->i and/c listof)
         (only-in racket/match match-define)
         (only-in racket/math pi sqr)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * + - / > clamp item log ne randn sum tensor? uniform! where
                  zeros)
         (only-in "../private/contract.rkt" define/checked-out supplied-or))

(provide uniform-bias)

(define dims/c (listof exact-nonnegative-integer?))

;; zeros + uniform! consumes the RNG exactly as torch's empty().uniform_().
(define/checked-out (uniform-init dims low high)
  (-> dims/c real? real? tensor?)
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
  (->* [dims/c] [#:a real?] tensor?)
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-init dims (- bound) bound))

;; the bias of nn.Linear and _ConvNd: U(-1/sqrt fan, 1/sqrt fan)
(define (uniform-bias n fan)
  (define bound (/ 1.0 (sqrt fan)))
  (uniform-init (list n) (- bound) bound))

(define (erf-series x)
  (define x² (* x x))
  (let loop ([n 0] [term x] [acc 0.0])
    (define next (+ acc (/ term (+ (* 2 n) 1))))
    (if (= next acc)
        (* (/ 2.0 (sqrt pi)) acc)
        (loop (add1 n) (/ (* term (- x²)) (add1 n)) next))))

(define (erfc-fraction x)
  (define k
    (for/fold ([k x]) ([n (in-range 80 0 -1)])
      (+ x (/ (/ n 2.0) k))))
  (/ (~> (* x x) - exp) (* (sqrt pi) k)))

(define (erf x)
  (cond
    [(< x 0) (~> x - erf -)]
    [(<= x 2.0) (erf-series x)]
    [else (- 1.0 (erfc-fraction x))]))

(define (any? mask)
  (~> mask sum item positive?))

(define (normal-draw dims mean std)
  (~> (apply randn dims) (* std) (+ mean)))

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
       [result tensor?])
  (define (cdf v)
    (/ (+ 1.0 (erf (/ (- v mean) std (sqrt 2.0)))) 2.0))
  (match-define (list μ σ lo hi) (map exact->inexact (list mean std a b)))
  (cond
    [(> (- (cdf b) (cdf a)) 0.3) (redrawn-normal dims μ σ lo hi)]
    [else (accepted-uniform dims μ σ lo hi)]))
