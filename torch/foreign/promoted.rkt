#lang racket/base

(require (only-in racket/list append* drop [flatten list-flatten] [take list-take])
         (only-in "device-type.rkt" device-type)
         (only-in "ops.rkt"
                  item tensor-device tensor-dtype tensor-shape tensor->list
                  to-device to-dtype)
         (only-in "size.rkt" ->2d)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" add mul reshape tensor unsqueeze)
         (prefix-in g: (only-in "../generated.rkt"
                                adaptive-avg-pool2d
                                avg-pool2d
                                conv2d
                                embedding
                                eq-scalar eq-tensor
                                gather
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
                                nonzero
                                select-int
                                slice-tensor
                                take
                                take-along-dim
                                tril
                                triu
                                where-scalarother
                                where-self)))

(provide flatten
         eq ne lt le gt ge
         conv2d max-pool2d avg-pool2d adaptive-avg-pool2d
         tril triu masked-fill embedding layer-norm
         gather take take-along-dim where
         tensor-ref :: slice? slice-start slice-end slice-step
         (rename-out [g:index-select index-select]
                     [g:masked-select masked-select]
                     [g:narrow narrow]
                     [g:nonzero nonzero]
                     [g:select-int select]))

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

;; Deviation from numpy advanced indexing: multiple index tensors
;; apply per-dim sequentially, not broadcast-combined, and index
;; tensors are rank-1 (contract).
(define (index-tensor positions v)
  (tensor positions #:dtype 'int64 #:device (tensor-device v)))

(define (mask-spec->index-tensor m)
  (reshape (g:nonzero (reshape m -1)) -1))

(define (wrap-negative-positions s v d)
  (define n (list-ref (tensor-shape v) d))
  (cond
    [(not (tensor? s))
     (define positions (if (vector? s) (vector->list s) s))
     (index-tensor (for/list ([i (in-list positions)])
                     (if (< i 0) (+ i n) i))
                   v)]
    [else
     ;; python indexing accepts a CPU index for a CUDA tensor;
     ;; index_select does not, so align devices first. Adding n*(s<0)
     ;; wraps negatives and is the identity elsewhere — no host sync.
     (define s* (to-device s (tensor-device v)))
     (add s* (mul (to-dtype (g:lt-scalar s* 0.0) 'int64)
                  (tensor n #:dtype 'int64
                          #:device (tensor-device v))))]))

(define (apply-mask-spec v d m0)
  (define m (to-device m0 (tensor-device v)))
  (define vdims (tensor-shape v))
  (define mdims (tensor-shape m))
  (define n (length mdims))
  (unless (and (<= (+ d n) (length vdims))
               (equal? mdims (list-take (list-tail vdims d) n)))
    (error 'tensor-ref "mask shape ~a does not match dims ~a at dim ~a"
           mdims vdims d))
  (define collapsed
    (if (= n 1)
        v
        (apply reshape v (append (list-take vdims d)
                                 (list (apply * mdims))
                                 (list-tail vdims (+ d n))))))
  (g:index-select collapsed d (mask-spec->index-tensor m)))

(define (tensor-ref t . specs)
  (cond
    [(and (pair? specs) (null? (cdr specs))
          (bool-mask? (car specs))
          (equal? (tensor-shape (car specs)) (tensor-shape t)))
     (g:masked-select t (to-device (car specs) (tensor-device t)))]
    [else
     (define expanded
       (expand-ellipsis specs (length (tensor-shape t)) 'tensor-ref))
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
           [(or (tensor? s) (list? s) (vector? s))
            (values (g:index-select v d (wrap-negative-positions s v d))
                    (add1 d))]
           [else (error 'tensor-ref "not an index spec: ~e" s)])))
     (cond
       [(pair? (tensor-shape result)) result]
       [(eq? (tensor-dtype result) 'bool) (not (zero? (item result)))]
       [else (item result)])]))

(define (take v n)
  (cond
    [(not (tensor? v)) (list-take v n)]
    [else
     (define idx (if (tensor? n) n (index-tensor n v)))
     (cond
       [(eq? (device-type (tensor-device v)) 'mps)
        ;; the vendored schema registers at::take for CPU/CUDA only, so
        ;; MPS composes flat index_select + reshape to the index shape;
        ;; index_select rejects the negatives native take wraps. The
        ;; same-device check keeps CPU/CUDA's strictness (the wrap's
        ;; to-device would otherwise silently migrate a mismatched idx)
        (unless (equal? (tensor-device idx) (tensor-device v))
          (error 'take
                 "index tensor must be on the same device as the input"))
        (define flat (reshape v -1))
        (define flat-out
          (g:index-select
           flat 0 (wrap-negative-positions (reshape idx -1) flat 0)))
        (apply reshape flat-out (tensor-shape idx))]
       [else (g:take v idx)])]))

(define (gather t dim index)
  (g:gather t dim index #f))

(define (take-along-dim t indices [dim #f])
  (g:take-along-dim t indices dim))

(define where
  (case-lambda
    [(mask)
     (cond
       [(null? (tensor-shape mask))
        (list (tensor (if (zero? (car (tensor->list mask))) '() '(0))
                      #:dtype 'int64 #:device (tensor-device mask)))]
       [else
        (define coords (g:nonzero mask))
        (for/list ([d (in-range (length (tensor-shape mask)))])
          (g:select-int coords 1 d))])]
    [(mask a b)
     (cond
       [(tensor? b) (g:where-self mask a b)]
       [(and (exact-integer? b) (memq (tensor-dtype a) '(bool int64)))
        ;; a double scalar would float-promote integral results and
        ;; round past 2^53 — same guard as the comparison combinator
        (g:where-self mask a (tensor b #:device (tensor-device a)))]
       [else (g:where-scalarother mask a (exact->inexact b))])]))

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
        (apply reshape v (append (list-take shp s)
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
