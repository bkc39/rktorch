#lang racket/base

(require (only-in racket/list append-map list-update make-list)
         (only-in racket/match match)
         (only-in "../foreign/raw/cas.rkt" cas-update!)
         (only-in (submod "param-group.rkt" private)
                  grouped? make-param-group param-group-settings param-group-tensors))

(provide group-store
         store-count
         store-groups
         store-initial-lrs!
         store-params
         store-partition
         store-ref
         store-set!
         store-set-all!
         store-set-each!)

(struct store (groups-box initial-lrs params owner))

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
  (store (box groups) (box (make-list (length groups) #f)) (append-map param-group-tensors groups)
         owner))

(define (store-groups st)
  (unbox (store-groups-box st)))

(define (store-count st)
  (length (store-groups st)))

(define (store-ref st i key)
  (hash-ref (param-group-settings (list-ref (store-groups st) i)) key))

(define (with-setting g key v)
  (make-param-group (param-group-tensors g) (hash-set (param-group-settings g) key (setting v))))

(define (update-groups! st f)
  (cas-update! (store-groups-box st) f)
  (void))

(define (store-set! st i key v)
  (update-groups! st (lambda (gs) (list-update gs i (lambda (g) (with-setting g key v))))))

(define (store-set-all! st key v)
  (update-groups! st (lambda (gs) (map (lambda (g) (with-setting g key v)) gs))))

(define (store-set-each! st key vs)
  (update-groups! st (lambda (gs) (map (lambda (g v) (with-setting g key v)) gs vs))))

(define (store-initial-lrs! st [initial #f])
  (define groups (store-groups st))
  (cas-update! (store-initial-lrs st)
               (lambda (initials)
                 (for/list ([g (in-list groups)] [recorded (in-list initials)])
                   (or initial recorded (hash-ref (param-group-settings g) 'lr))))))

(define (store-partition st qs gs)
  (match (store-groups st)
    [(list g) (list (list (param-group-settings g) qs gs))]
    [groups
     (define owner (store-owner st))
     (for/list ([(g i) (in-indexed groups)]
                #:do [(define-values (oqs ogs) (owned-by owner i qs gs))]
                #:unless (null? oqs))
       (list (param-group-settings g) oqs ogs))]))

(define (owned-by owner i qs gs)
  (for/lists (oqs ogs) ([q (in-list qs)] [g (in-list gs)]
                        #:when (eqv? (hash-ref owner q) i))
    (values q g)))
