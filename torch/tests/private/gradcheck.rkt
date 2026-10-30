#lang racket/base

(require (only-in ffi/vector list->f64vector)
         (only-in racket/list list-set)
         (only-in threading ~>)
         (only-in "../../foreign.rkt"
                  * backward! detach grad item make-generator randn reshape
                  requires-grad! sum tensor tensor->list tensor-shape to-dtype))

(provide gradcheck)

(define (flat x)
  (~> x (reshape -1) tensor->list))

(define (shaped-like x values)
  (apply reshape (tensor (list->f64vector values)) (tensor-shape x)))

(define (gradcheck f inputs
                   #:eps [eps 1e-6] #:atol [atol 1e-5] #:rtol [rtol 1e-3])
  (define xs (map (lambda (x) (~> x detach (to-dtype 'float64))) inputs))
  (define w (randn (tensor-shape (apply f xs)) #:dtype 'float64
                   #:generator (make-generator 0)))
  (define (loss args)
    (~> (apply f args) (* w) sum))
  (define leaves (map (lambda (x) (requires-grad! (detach x))) xs))
  (backward! (loss leaves))
  (for/and ([x (in-list xs)] [leaf (in-list leaves)] [k (in-naturals)])
    (define entries (flat x))
    (for/and ([analytic (in-list (flat (grad leaf)))]
              [entry (in-list entries)]
              [i (in-naturals)])
      (define (moved delta)
        (~> xs
            (list-set k (shaped-like x (list-set entries i (+ entry delta))))
            loss
            item))
      (define numeric (/ (- (moved eps) (moved (- eps))) (* 2 eps)))
      (<= (abs (- analytic numeric)) (+ atol (* rtol (abs numeric)))))))
