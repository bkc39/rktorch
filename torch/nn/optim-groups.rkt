#lang racket/base

(require (only-in racket/list append-map)
         (only-in (submod "param-group.rkt" private)
                  grouped? make-param-group param-group-settings param-group-tensors))

(provide group-store
         set-store-initial-lrs!
         store-count
         store-groups
         store-initial-lrs!
         store-params
         store-partition
         store-ref
         store-set!
         store-set-all!
         store-set-each!)

(struct store ([groups #:mutable] [initial-lrs #:mutable] params owner))

(define (setting v)
  (if (boolean? v) v (real->double-flonum v)))

(define (filled defaults g)
  (make-param-group (param-group-tensors g)
                    (for/fold ([h defaults]) ([(k v) (in-hash (param-group-settings g))])
                      (hash-set h k (setting v)))))

(define (group-store params defaults)
  (define base
    (for/hasheq ([(k v) (in-hash defaults)])
      (values k (setting v))))
  (define groups
    (if (grouped? params)
        (map (lambda (g) (filled base g)) params)
        (list (make-param-group params base))))
  (define owner
    (for*/hasheq ([(g i) (in-indexed groups)]
                  [p (in-list (param-group-tensors g))])
      (values p i)))
  (store groups (map (lambda (_) #f) groups) (append-map param-group-tensors groups) owner))

(define (in-indexed xs)
  (in-parallel (in-list xs) (in-naturals)))

(define (store-count st)
  (length (store-groups st)))

(define (store-ref st i key)
  (hash-ref (param-group-settings (list-ref (store-groups st) i)) key))

(define (with-setting g key v)
  (make-param-group (param-group-tensors g) (hash-set (param-group-settings g) key (setting v))))

(define (store-set! st i key v)
  (set-store-groups! st (for/list ([(g j) (in-indexed (store-groups st))])
                          (if (= i j) (with-setting g key v) g))))

(define (store-set-all! st key v)
  (set-store-groups! st (for/list ([g (in-list (store-groups st))])
                          (with-setting g key v))))

(define (store-set-each! st key vs)
  (set-store-groups! st (for/list ([g (in-list (store-groups st))] [v (in-list vs)])
                          (with-setting g key v))))

(define (store-initial-lrs! st)
  (define lrs
    (for/list ([g (in-list (store-groups st))] [initial (in-list (store-initial-lrs st))])
      (or initial (hash-ref (param-group-settings g) 'lr))))
  (set-store-initial-lrs! st lrs)
  lrs)

(define (store-partition st qs gs)
  (define groups (store-groups st))
  (cond
    [(null? (cdr groups)) (list (list (param-group-settings (car groups)) qs gs))]
    [else
     (define owner (store-owner st))
     (for*/list ([(g i) (in-indexed groups)]
                 [part (in-value (owned-by owner i qs gs))]
                 #:unless (null? (car part)))
       (cons (param-group-settings g) part))]))

(define (owned-by owner i qs gs)
  (define-values (oqs ogs)
    (for/lists (oqs ogs) ([q (in-list qs)] [g (in-list gs)]
                          #:when (eqv? (hash-ref owner q) i))
      (values q g)))
  (list oqs ogs))
