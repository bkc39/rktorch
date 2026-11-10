#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     ;; whole-module on purpose: the syntax classes need
                     ;; bindings only-in would strip
                     syntax/parse/pre)
         (only-in racket/contract/base
                  -> ->i and/c contract-out flat-contract-predicate
                  flat-named-contract listof)
         (only-in racket/match match-define)
         (only-in racket/math nan? sqr)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * + - / > clamp item log ne normal! sum tensor? uniform!
                  where zeros)
         (only-in "../foreign/generator.rkt"
                  check-generator-on! generator-on? generator/c range-message
                  uniform-range?)
         (only-in "../private/contract.rkt" define/checked-out supplied-or)
         (only-in "../private/erf.rkt" erf))

(provide uniform-bias)

(define dims/c (listof exact-nonnegative-integer?))

(define dims? (flat-contract-predicate dims/c))

(define fan-dims/c
  (flat-named-contract 'fan-dims
                       (lambda (d) (and (dims? d) (>= (length d) 2)))))

(define slope/c
  (flat-named-contract 'non-nan-real
                       (lambda (a) (and (real? a) (not (nan? a))))))

(define (init-range? low high)
  (uniform-range? 'float32 low high))

(begin-for-syntax
  (define-splicing-syntax-class formal
    #:attributes (name (spec 1) (arg 1))
    (pattern name:id
             #:with (spec ...) #'(name)
             #:with (arg ...) #'(name))
    (pattern (~seq kw:keyword [name:id default:expr])
             #:with (spec ...) #'(kw [name default])
             #:with (arg ...) #'(kw name)))

  (define-splicing-syntax-class precondition
    #:attributes ((clause 1))
    (pattern (~seq #:pre/name deps message:str check:expr)
             #:with (clause ...) #'(#:pre/name deps message check))
    (pattern (~seq #:pre/name deps
                   (#:range low:str high:str (~optional condition:str))
                   check:expr)
             #:with message
             (datum->syntax
              #'check
              (string-append
               (range-message (syntax-e #'low) (syntax-e #'high) "float32's")
               (if (attribute condition)
                   (string-append " when " (syntax-e #'condition))
                   "")))
             #:with (clause ...) #'(#:pre/name deps message check))))

(define-syntax-parse-rule (define-init (name:id formal:formal ...)
                            ((~literal ->i) (mandatory ...) (optional ...)
                                            pre:precondition ...
                                            result)
                            body:expr ...+)
  #:with generator (format-id #'name "generator")
  (begin
    (define contracted
      (let ()
        (define (name formal.spec ... ... #:generator [generator #f])
          body ...)
        name))
    (define (name formal.spec ... ... #:generator [generator #f])
      (check-generator-on! 'name generator #f)
      (contracted formal.arg ... ... #:generator generator))
    (provide name)
    (module+ checked
      (provide
       (contract-out
        [rename contracted name
                (->i (mandatory ...)
                     (optional ... #:generator [generator generator/c])
                     pre.clause ... ...
                     #:pre/name (generator)
                     "the generator lives on the target device"
                     (generator-on? (supplied-or generator) #f)
                     result)])))))

(define (uniform-draw dims low high generator)
  (define t (apply zeros dims))
  (uniform! t low high #:generator generator)
  t)

(define (normal-draw dims mean std generator)
  (define t (apply zeros dims))
  (normal! t mean std #:generator generator)
  t)

(define-init (uniform-init dims low high) ;; noqa
  (->i ([dims dims/c] [low real?] [high real?])
       ()
       #:pre/name (low high) (#:range "low" "high") (init-range? low high)
       [result tensor?])
  (uniform-draw dims low high generator))

(define-init (normal-init dims) ;; noqa
  (->i ([dims dims/c]) () [result tensor?])
  (normal-draw dims 0.0 1.0 generator))

(define/checked-out (fan-in dims)
  (-> dims/c exact-nonnegative-integer?)
  (apply * (cdr dims)))

(define-init (kaiming-uniform dims #:a [a (sqrt 5.0)]) ;; noqa
  (->i ([dims fan-dims/c]) (#:a [a slope/c]) [result tensor?])
  (define gain (sqrt (/ 2.0 (+ 1.0 (* a a)))))
  (cond
    [(memv 0 dims) (apply zeros dims)]
    [else
     (define bound (* (sqrt 3.0) (/ gain (sqrt (fan-in dims)))))
     (uniform-draw dims (- bound) bound generator)]))

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

(define-init (trunc-normal-init dims ;; noqa
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
       (#:range "#:a" "#:b" "[a, b] holds at most 0.3 of the mass")
       (let ([a (supplied-or a -2.0)] [b (supplied-or b 2.0)])
         (or (not (uniform-branch? (supplied-or mean 0.0) (supplied-or std 1.0)
                                   a b))
             (init-range? a b)))
       [result tensor?])
  (match-define (list μ σ lo hi) (map exact->inexact (list mean std a b)))
  (cond
    [(uniform-branch? mean std a b) (accepted-uniform dims μ σ lo hi generator)]
    [else (redrawn-normal dims μ σ lo hi generator)]))
