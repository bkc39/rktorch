#lang racket/base

(require (only-in racket/dict dict-values in-dict)
         (only-in racket/match match match* match-lambda)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "layer-containers.rkt" Children% Children%? Parameters%)
         (only-in "layer-core.rkt" fields-of fields? layer-named-children layer?)
         (only-in "memo.rkt" memo remembered))

(provide rebuilt-values
         refuse-cycles)

(define (node? v)
  (or (layer? v) (Children%? v) (pair? v) (vector? v) (box? v) (hash? v)))

(define (entries-of m)
  (append (if (fields? m) ((fields-of m) m) '()) (layer-named-children m)))

(define (children-of v field)
  (define (labelled parts)
    (map (lambda (x) (cons field x)) parts))
  (match v
    [(? layer?) (entries-of v)]
    [(Children% alist) (labelled (dict-values alist))]
    [(cons a d) (labelled (list a d))]
    [(? vector?) (labelled (vector->list v))]
    [(box x) (labelled (list x))]
    [(? hash?)
     (labelled (for*/list ([(k x) (in-hash v)] [part (in-list (list k x))])
                 part))]))

(define (refuse-cycles who m)
  (define seen (memo who))
  (define (visit-all entries)
    (for ([(name x) (in-dict entries)])
      (visit name x)))
  (define (visit field v)
    (when (node? v)
      (remembered seen v (lambda () (visit-all (children-of v field))) #:field field)))
  (remembered seen m (lambda () (visit-all (entries-of m)))))

(define (rebuilt-values seen names vals child tensor)
  (for/list ([name (in-list names)] [v (in-list vals)])
    (rebuilt-value seen name v child tensor)))

(define (rebuilt-value seen name v child tensor)
  (let walk ([v v])
    (match v
      [(? layer?) (child name v)]
      [(? tensor?) (tensor v)]
      [(Parameters% alist)
       (Parameters% (map (match-lambda [(cons k t) (cons k (tensor t))]) alist))]
      [(Children% alist)
       (Children% (map (match-lambda [(cons k c) (cons k (child k c))]) alist))]
      [(? node?)
       (remembered seen v
                   (lambda () (reassembled v (dict-values (children-of v name)) walk))
                   #:field name)]
      [_ v])))

(define (reassembled v parts walk)
  (match* (v parts)
    [((? pair?) (list a d)) (cons (walk a) (walk d))]
    [((? vector?) _) (apply (if (immutable? v) vector-immutable vector) (map walk parts))]
    [((? box?) (list x)) ((if (immutable? v) box-immutable box) (walk x))]
    [((? hash?) _)
     (refilled (hash-copy-clear v) parts (if (weakly-keyed? v) values walk) walk)]))

(define (weakly-keyed? h)
  (or (hash-weak? h) (hash-ephemeron? h)))

(define (refilled h parts key walk)
  (match parts
    ['() h]
    [(list* k x more) (refilled (hash-put h (key k) (walk x)) more key walk)]))

(define (hash-put h k x)
  (cond
    [(immutable? h) (hash-set h k x)]
    [else
     (hash-set! h k x)
     h]))
