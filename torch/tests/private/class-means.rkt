#lang racket/base

(require (only-in racket/list argmin)
         (only-in racket/math sqr))

(provide class-means
         nearest-class
         squared-distance
         cosine-distance)

(define (class-means features labels)
  (for/fold ([sums (hash)] [counts (hash)]
             #:result (for/hash ([(k sum) (in-hash sums)])
                        (values k (map (lambda (s) (/ s (hash-ref counts k))) sum))))
            ([f (in-list features)] [k (in-list labels)])
    (values (hash-update sums k (lambda (s) (map + s f)) (map (lambda (_) 0) f))
            (hash-update counts k add1 0))))

(define (nearest-class means feature distance)
  (argmin (lambda (k) (distance feature (hash-ref means k))) (hash-keys means)))

(define (squared-distance a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (sqr (- x y))))

(define (dot a b)
  (for/sum ([x (in-list a)] [y (in-list b)]) (* x y)))

(define (cosine-distance a b)
  (- 1 (/ (dot a b) (sqrt (* (dot a a) (dot b b))))))
