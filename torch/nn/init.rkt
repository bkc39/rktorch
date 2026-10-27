#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i and/c listof unsupplied-arg?)
         (only-in racket/math pi)
         (only-in threading ~>)
         (only-in "../foreign.rkt" * + clamp randn tensor? uniform! zeros)
         (only-in "../generated.rkt" erfinv)
         (only-in "../private/contract.rkt" define/checked-out))

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
  (let loop ([n 0] [term x] [sum 0.0])
    (define next (+ sum (/ term (+ (* 2 n) 1))))
    (if (= next sum)
        (* (/ 2.0 (sqrt pi)) sum)
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

(define (supplied-or v default)
  (if (unsupplied-arg? v) default v))

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
  (~> (uniform-init dims (- (* 2.0 (cdf a)) 1.0) (- (* 2.0 (cdf b)) 1.0))
      erfinv
      (* (* std (sqrt 2.0)))
      (+ (exact->inexact mean))
      (clamp #:min a #:max b)))
