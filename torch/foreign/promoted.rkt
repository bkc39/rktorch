#lang racket/base

(require (only-in racket/list append* drop [flatten list-flatten] take)
         (only-in "ops.rkt"
                  item tensor->list tensor-device tensor-dtype tensor-shape)
         (only-in "size.rkt" ->2d)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" reshape tensor unsqueeze)
         (prefix-in g: (only-in "../generated.rkt"
                                adaptive-avg-pool2d
                                avg-pool2d
                                conv2d
                                embedding
                                eq-scalar eq-tensor
                                ge-scalar ge-tensor
                                gt-scalar gt-tensor
                                index-select
                                layer-norm
                                le-scalar le-tensor
                                lt-scalar lt-tensor
                                masked-fill-scalar
                                masked-select
                                max-pool2d
                                narrow
                                ne-scalar ne-tensor
                                select-int
                                slice-tensor
                                tril
                                triu)))

(provide flatten
         eq ne lt le gt ge
         conv2d max-pool2d avg-pool2d adaptive-avg-pool2d
         tril triu masked-fill embedding layer-norm
         ref :: slice?
         (rename-out [ref tensor-ref]
                     [g:narrow narrow]
                     [g:select-int select]))

;; :: follows python slice()'s argument convention: (::) is [:],
;; (:: n) is [:n], (:: a b) is [a:b], (:: a b s) is [a:b:s]; #f leaves
;; a bound open.
(struct slice (start end step) #:transparent)
(define ::
  (case-lambda
    [() (slice #f #f 1)]
    [(end) (slice #f end 1)]
    [(start end) (slice start end 1)]
    [(start end step) (slice start end step)]))

(define (bool-mask? s)
  (and (tensor? s) (eq? (tensor-dtype s) 'bool)))

(define (spec-consumed-dims s)
  (cond
    [(or (eq? s '...) (eq? s #f)) 0]
    [(bool-mask? s) (length (tensor-shape s))]
    [else 1]))

(define (expand-ellipsis specs rank who)
  (define consuming
    (for/sum ([s (in-list specs)]) (spec-consumed-dims s)))
  (unless (<= consuming rank)
    (error who "too many indices for a ~a-d tensor: ~e" rank specs))
  (define fills (- rank consuming))
  (define ellipses (for/sum ([s (in-list specs)]) (if (eq? s '...) 1 0)))
  (case ellipses
    [(0) specs]
    [(1) (append* (for/list ([s (in-list specs)])
                    (if (eq? s '...) (build-list fills (lambda (_) (::)))
                        (list s))))]
    [else (error who "at most one '... allowed: ~e" specs)]))

;; The python indexing surface, spec-for-spec: integers select (rank
;; drops; a fully-indexed result auto-items to a scalar, booleans as
;; #t/#f), slices/'...' stay views, #f is None (new axis), an int list
;; or rank-1 int64 tensor is index_select along that dim (negative
;; positions wrap, as in python), and a bool tensor consumes as many
;; dims as its rank — python's semantics: the masked dims collapse to
;; one true-count dim. A full-rank bool tensor alone stays
;; device-resident via masked_select. Deviation from numpy advanced
;; indexing: multiple index tensors apply per-dim sequentially, not
;; broadcast-combined, and index tensors are rank-1 (contract).
(define (index-tensor positions v)
  (tensor positions #:dtype 'int64 #:device (tensor-device v)))

(define (mask-spec->index-tensor m v)
  (index-tensor (for/list ([x (in-list (tensor->list m))]
                           [i (in-naturals)]
                           #:when (not (zero? x)))
                  i)
                v))

(define (wrap-negative-positions s v d)
  (define n (list-ref (tensor-shape v) d))
  (define positions
    (if (list? s)
        s
        (tensor->list s)))
  (cond
    [(ormap negative? positions)
     (index-tensor (for/list ([i (in-list positions)])
                     (if (< i 0) (+ i n) i))
                   v)]
    [(list? s) (index-tensor positions v)]
    [else s]))

(define (apply-mask-spec v d m)
  (define vdims (tensor-shape v))
  (define mdims (tensor-shape m))
  (define n (length mdims))
  (unless (and (<= (+ d n) (length vdims))
               (equal? mdims (take (list-tail vdims d) n)))
    (error 'ref "mask shape ~a does not match dims ~a at dim ~a"
           mdims vdims d))
  (define collapsed
    (if (= n 1)
        v
        (apply reshape v (append (take vdims d)
                                 (list (apply * mdims))
                                 (list-tail vdims (+ d n))))))
  (g:index-select collapsed d (mask-spec->index-tensor m collapsed)))

(define (ref t . specs)
  (cond
    [(and (pair? specs) (null? (cdr specs))
          (bool-mask? (car specs))
          (equal? (tensor-shape (car specs)) (tensor-shape t)))
     (g:masked-select t (car specs))]
    [else
     (define expanded
       (expand-ellipsis specs (length (tensor-shape t)) 'ref))
     (define result
       (for/fold ([v t] [d 0] #:result v) ([s (in-list expanded)])
         (cond
           [(exact-integer? s) (values (g:select-int v d s) d)]
           [(slice? s)
            (values (g:slice-tensor v d (slice-start s) (slice-end s)
                                    (slice-step s))
                    (add1 d))]
           [(eq? s #f) (values (unsqueeze v d) (add1 d))]
           [(bool-mask? s) (values (apply-mask-spec v d s) (add1 d))]
           [(or (tensor? s) (list? s))
            (values (g:index-select v d (wrap-negative-positions s v d))
                    (add1 d))]
           [else (error 'ref "not an index spec: ~e" s)])))
     (cond
       [(pair? (tensor-shape result)) result]
       [(eq? (tensor-dtype result) 'bool) (not (zero? (item result)))]
       [else (item result)])]))

(define (flatten v [start-dim 0] [end-dim -1])
  (cond
    [(tensor? v)
     (define shp (tensor-shape v))
     (define n (length shp))
     (cond
       [(zero? n)
        (unless (and (memv start-dim '(0 -1)) (memv end-dim '(0 -1)))
          (error 'flatten
                 "invalid dim range [~a, ~a] for a 0-d tensor"
                 start-dim end-dim))
        (reshape v 1)]
       [else
        (define s (if (negative? start-dim) (+ n start-dim) start-dim))
        (define e (if (negative? end-dim) (+ n end-dim) end-dim))
        (unless (and (<= 0 s e) (< e n))
          (error 'flatten
                 "invalid dim range [~a, ~a] for a ~a-d tensor"
                 start-dim end-dim n))
        ;; not for/product — that would raise the racket/base version floor
        (define collapsed
          (apply * (for/list ([d (in-list shp)] [i (in-naturals)]
                                                 #:when (<= s i e))
                     d)))
        (apply reshape v (append (take shp s)
                                 (list collapsed)
                                 (drop shp (add1 e))))])]
    [else (list-flatten v)]))

;; --------------------------------------------------- comparison dispatchers

(define ((comparison t-op s-op) a b)
  (cond
    [(tensor? b) (t-op a b)]
    [(and (exact-integer? b) (eq? (tensor-dtype a) 'int64))
     ;; the scalar path transits a C double, which rounds past 2^53; the
     ;; rhs tensor must live on a's device, not the default
     (t-op a (tensor b #:device (tensor-device a)))]
    [else (s-op a (exact->inexact b))]))

(define eq (comparison g:eq-tensor g:eq-scalar))
(define ne (comparison g:ne-tensor g:ne-scalar))
(define lt (comparison g:lt-tensor g:lt-scalar))
(define le (comparison g:le-tensor g:le-scalar))
(define gt (comparison g:gt-tensor g:gt-scalar))
(define ge (comparison g:ge-tensor g:ge-scalar))

;; --------------------------------------------------- conv + pooling wrappers

(define (conv2d input weight
                #:bias [bias #f]
                #:stride [stride 1]
                #:padding [padding 0]
                #:dilation [dilation 1]
                #:groups [groups 1])
  (g:conv2d input weight bias
            (->2d stride) (->2d padding) (->2d dilation) groups))

(define (max-pool2d input kernel-size
                    #:stride [stride #f]
                    #:padding [padding 0]
                    #:dilation [dilation 1]
                    #:ceil-mode [ceil-mode #f])
  (g:max-pool2d input
                (->2d kernel-size)
                (->2d (or stride kernel-size))
                (->2d padding) (->2d dilation) ceil-mode))

(define (avg-pool2d input kernel-size
                    #:stride [stride #f]
                    #:padding [padding 0]
                    #:ceil-mode [ceil-mode #f]
                    #:count-include-pad [count-include-pad #t]
                    #:divisor-override [divisor-override #f])
  (g:avg-pool2d input
                (->2d kernel-size)
                (->2d (or stride kernel-size))
                (->2d padding) ceil-mode count-include-pad divisor-override))

(define (adaptive-avg-pool2d input output-size)
  (g:adaptive-avg-pool2d input (->2d output-size)))

;; ------------------------------------------- transformer primitives

(define (tril self [diagonal 0])
  (g:tril self diagonal))

(define (triu self [diagonal 0])
  (g:triu self diagonal))

(define (masked-fill self mask value)
  (g:masked-fill-scalar self mask (exact->inexact value)))

(define (embedding indices weight #:padding-idx [padding-idx #f])
  (g:embedding weight indices (or padding-idx -1) #f #f))

(define (layer-norm input normalized-shape
                    #:weight [weight #f]
                    #:bias [bias #f]
                    #:eps [eps 1e-5])
  (define shape
    (if (list? normalized-shape) normalized-shape (list normalized-shape)))
  (g:layer-norm input shape weight bias eps #t))
