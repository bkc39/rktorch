#lang racket/base

(require (only-in racket/contract/base
                  -> ->i and/c flat-contract-predicate flat-named-contract listof
                  or/c)
         (only-in racket/match match-define)
         (only-in racket/math sqr)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * + - / > clamp item log ne normal! sum tensor? uniform!
                  where zeros)
         (only-in "../foreign/generator.rkt"
                  check-generator-on! generator? generator-on? uniform-range?)
         (only-in "../private/contract.rkt" define/checked-out supplied-or)
         (only-in "../private/erf.rkt" erf))

(provide uniform-bias)

(define dims/c (listof exact-nonnegative-integer?))

(define dims? (flat-contract-predicate dims/c))

(define fan-dims/c
  (flat-named-contract 'fan-dims
                       (lambda (d) (and (dims? d) (>= (length d) 2)))))

(define (uniform-draw dims low high generator)
  (define t (apply zeros dims))
  (uniform! t low high #:generator generator)
  t)

(define (normal-draw dims mean std generator)
  (define t (apply zeros dims))
  (normal! t mean std #:generator generator)
  t)

(define/checked-out (uniform-init dims low high #:generator [generator #f])
  (->i ([dims dims/c] [low real?] [high real?])
       (#:generator [generator (or/c generator? #f)])
       #:pre/name (generator) "the generator lives on the target device"
       (generator-on? (supplied-or generator) #f)
       #:pre/name (low high) "low is at most high, within float32's finite range"
       (uniform-range? 'float32 low high)
       [result tensor?])
  (check-generator-on! 'uniform-init generator #f)
  (uniform-draw dims low high generator))

(define/checked-out (normal-init dims #:generator [generator #f]) ;; noqa
  (->i ([dims dims/c])
       (#:generator [generator (or/c generator? #f)])
       #:pre/name (generator) "the generator lives on the target device"
       (generator-on? (supplied-or generator) #f)
       [result tensor?])
  (check-generator-on! 'normal-init generator #f)
  (normal-draw dims 0.0 1.0 generator))

(define/checked-out (fan-in dims)
  (-> dims/c exact-nonnegative-integer?)
  (apply * (cdr dims)))

(define/checked-out (kaiming-uniform dims ;; noqa
                                     #:a [a (sqrt 5.0)]
                                     #:generator [generator #f])
  (->i ([dims fan-dims/c])
       (#:a [a real?] #:generator [generator (or/c generator? #f)])
       #:pre/name (generator) "the generator lives on the target device"
       (generator-on? (supplied-or generator) #f)
       [result tensor?])
  (check-generator-on! 'kaiming-uniform generator #f)
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
  (uniform-draw dims (- bound) bound generator))

(define (uniform-bias n fan #:generator [generator #f])
  (check-generator-on! 'uniform-bias generator #f)
  (define bound (/ 1.0 (sqrt fan)))
  (uniform-draw (list n) (- bound) bound generator))

(define (any? mask)
  (~> mask sum item positive?))

(define (redrawn-normal dims mean std a b generator)
  (let loop ([result (normal-draw dims mean std generator)])
    (define outside (ne result (clamp result #:min a #:max b)))
    (cond
      [(any? outside)
       (loop (where outside (normal-draw dims mean std generator) result))]
      [else result])))

(define (accepted-uniform dims mean std a b generator)
  (define log-peak (~> (min mean b) (max a) (- mean) (/ std) sqr (* -0.5)))
  (define (rejected candidates)
    (define z (/ (- candidates mean) std))
    (define log-density (- (* z z -0.5) log-peak))
    (~> (uniform-draw dims 0.0 1.0 generator) log (> log-density)))
  (define drawn (uniform-draw dims a b generator))
  (let loop ([result drawn] [pending (rejected drawn)])
    (cond
      [(any? pending)
       (define candidates (uniform-draw dims a b generator))
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
                                       #:b [b 2.0]
                                       #:generator [generator #f])
  (->i ([dims dims/c])
       (#:mean [mean real?]
        #:std [std (and/c real? positive?)]
        #:a [a real?]
        #:b [b real?]
        #:generator [generator (or/c generator? #f)])
       #:pre/name (a b) "#:a is below #:b"
       (< (supplied-or a -2.0) (supplied-or b 2.0))
       #:pre/name (mean std a b)
       "#:a and #:b lie within float32's finite range when [a, b] holds at most 0.3 of the mass"
       (let ([a (supplied-or a -2.0)] [b (supplied-or b 2.0)])
         (or (not (uniform-branch? (supplied-or mean 0.0) (supplied-or std 1.0)
                                   a b))
             (uniform-range? 'float32 a b)))
       #:pre/name (generator) "the generator lives on the target device"
       (generator-on? (supplied-or generator) #f)
       [result tensor?])
  (check-generator-on! 'trunc-normal-init generator #f)
  (match-define (list μ σ lo hi) (map exact->inexact (list mean std a b)))
  (cond
    [(uniform-branch? mean std a b) (accepted-uniform dims μ σ lo hi generator)]
    [else (redrawn-normal dims μ σ lo hi generator)]))
