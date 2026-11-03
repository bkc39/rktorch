#lang racket/base

(require (only-in racket/dict in-dict in-dict-values)
         (only-in racket/match == match match-lambda)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "layer-containers.rkt" Children% Children%? Parameters%)
         (only-in "layer-core.rkt" fields-of fields? layer-named-children layer?))

(provide copy-child
         rebuilt-values
         refuse-cycles)

(struct copy-child (memo proc)
  #:property prop:procedure (struct-field-index proc))

(define building (string->uninterned-symbol "building"))

(define (node? v)
  (or (layer? v) (Children%? v) (pair? v) (vector? v) (box? v) (hash? v)))

(define (entries-of m)
  (append (if (fields? m) ((fields-of m) m) '()) (layer-named-children m)))

(define (refuse-cycles m)
  (define open (make-hasheq))
  (define done (make-hasheq))
  (let visit ([owner #f] [field #f] [v m])
    (cond
      [(not (node? v)) (void)]
      [(hash-ref open v #f)
       (raise-arguments-error 'layer-copy
                              "a field's value reaches itself, and a cycle cannot be copied"
                              "field" field
                              "layer" owner)]
      [(hash-ref done v #f) (void)]
      [else
       (hash-set! open v #t)
       (match v
         [(? layer?)
          (for ([(name x) (in-dict (entries-of v))])
            (visit v name x))]
         [(Children% alist)
          (for ([x (in-dict-values alist)])
            (visit owner field x))]
         [(cons a d)
          (visit owner field a)
          (visit owner field d)]
         [(? vector?)
          (for ([x (in-vector v)])
            (visit owner field x))]
         [(box x) (visit owner field x)]
         [(? hash?)
          (for ([(k x) (in-hash v)])
            (visit owner field k)
            (visit owner field x))])
       (hash-remove! open v)
       (hash-set! done v #t)])))

(define (rebuilt-values names vals child tensor) ;; noqa
  (define memo (if (copy-child? child) (copy-child-memo child) (make-hasheq)))
  (for/list ([name (in-list names)] [v (in-list vals)])
    (rebuilt-value memo name v child tensor)))

(define (rebuilt-value memo name v child tensor)
  (define (shared original build)
    (match (hash-ref memo original #f)
      [#f
       (hash-set! memo original building)
       (define copy (build))
       (hash-set! memo original copy)
       copy]
      [(== building eq?)
       (raise-arguments-error 'layer-rebuild
                              "a field's value reaches itself, and a cycle cannot be rebuilt"
                              "field" name)]
      [copy copy]))
  (let walk ([v v])
    (match v
      [(? layer?) (child name v)]
      [(? tensor?) (tensor v)]
      [(Parameters% alist)
       (Parameters% (map (match-lambda [(cons k t) (cons k (tensor t))]) alist))]
      [(Children% alist)
       (Children% (map (match-lambda [(cons k c) (cons k (child k c))]) alist))]
      [(cons a d) (shared v (lambda () (cons (walk a) (walk d))))]
      [(? vector?) (shared v (lambda () (rebuilt-vector v walk)))]
      [(box x)
       (shared v (lambda () ((if (immutable? v) box-immutable box) (walk x))))]
      [(? hash?) (shared v (lambda () (rebuilt-hash v walk)))]
      [_ v])))

(define (rebuilt-vector v walk)
  (define copy (for/vector #:length (vector-length v) ([x (in-vector v)])
                 (walk x)))
  (if (immutable? v) (vector->immutable-vector copy) copy))

(define (rebuilt-hash h walk)
  (define-values (ks xs)
    (for/lists (ks xs) ([(k x) (in-hash h)])
      (values (walk k) (walk x))))
  (cond
    [(immutable? h)
     (for/fold ([acc (hash-copy-clear h)]) ([k (in-list ks)] [x (in-list xs)])
       (hash-set acc k x))]
    [else
     (define copy (hash-copy-clear h))
     (for ([k (in-list ks)] [x (in-list xs)])
       (hash-set! copy k x))
     copy]))
