#lang racket/base

(require (only-in racket/base [sort base:sort])
         (only-in racket/contract/base
                  -> ->* ->i and/c any any/c flat-named-contract none/c or/c
                  unsupplied-arg?)
         (prefix-in g: (only-in "../generated.rkt"
                                argsort multinomial sort-tensor topk))
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "contracts.rkt" index/c)
         (only-in "creation-ops.rkt" generator?)
         (only-in "ops.rkt" tensor-shape)
         (only-in "structs.rkt" tensor?))

(define (supplied v default)
  (if (unsupplied-arg? v) default v))

(define sort/c
  (->i ([v (or/c tensor? list?)])
       ([less-than? (v) (if (tensor? v) none/c (any/c any/c . -> . any/c))]
        #:key [key (v) (if (tensor? v) none/c (or/c #f (any/c . -> . any/c)))]
        #:cache-keys? [cache-keys? (v) (if (tensor? v) none/c boolean?)]
        #:dim [dim (v) (if (tensor? v) index/c none/c)]
        #:descending? [descending? (v) (if (tensor? v) boolean? none/c)])
       #:pre/name (v less-than?) "a list is sorted by a less-than? procedure"
       (or (tensor? v) (not (unsupplied-arg? less-than?)))
       any))

(define/contract-out (sort v [less-than? #f] ;; noqa
                           #:key [key #f]
                           #:cache-keys? [cache-keys? #f]
                           #:dim [dim -1]
                           #:descending? [descending? #f])
  sort/c
  (cond
    [(tensor? v) (g:sort-tensor v dim descending?)]
    [key (base:sort v less-than? #:key key #:cache-keys? cache-keys?)]
    [else (base:sort v less-than?)]))

(define/contract-out (argsort t #:dim [dim -1] #:descending? [descending? #f]) ;; noqa
  (->* [tensor?] [#:dim index/c #:descending? boolean?] tensor?)
  (g:argsort t dim descending?))

(define (dim-length t dim)
  (define shape (tensor-shape t))
  (define rank (length shape))
  (define axis (if (negative? dim) (+ dim rank) dim))
  (cond
    [(null? shape) 1]
    [(< -1 axis rank) (list-ref shape axis)]
    [else #f]))

(define topk/c
  (->i ([t tensor?]
        [k exact-nonnegative-integer?])
       (#:dim [dim index/c]
        #:largest? [largest? boolean?]
        #:sorted? [sorted? boolean?])
       #:pre/name (t k dim) "k is at most the length of dim"
       (let ([n (dim-length t (supplied dim -1))])
         (or (not n) (<= k n)))
       (values [top tensor?] [indices tensor?])))

(define/contract-out (topk t k ;; noqa
                           #:dim [dim -1]
                           #:largest? [largest? #t]
                           #:sorted? [sorted? #t])
  topk/c
  (g:topk t k dim largest? sorted?))

(define probabilities/c
  (flat-named-contract
   'probabilities
   (and/c tensor? (lambda (t) (<= 1 (length (tensor-shape t)) 2)))))

(define/contract-out (multinomial probabilities num-samples ;; noqa
                                  #:replacement? [replacement? #f]
                                  #:generator [generator #f])
  (->* [probabilities/c exact-positive-integer?]
       [#:replacement? boolean? #:generator (or/c generator? #f)]
       tensor?)
  (g:multinomial probabilities num-samples replacement? generator))
