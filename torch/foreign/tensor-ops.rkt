#lang racket/base

(require (only-in ffi/vector list->s64vector)
         (only-in racket/base
                  [exp base:exp]
                  [log base:log]
                  [max base:max]
                  [min base:min]
                  [sqrt base:sqrt])
         (only-in racket/contract/base -> ->* non-empty-listof)
         (only-in racket/list [argmax base:argmax])
         (only-in racket/math [tanh base:tanh])
         (only-in "../private/contract.rkt"
                  define/checked-out
                  define/contract-out)
         (only-in "contracts.rkt"
                  argmax/c
                  binary-arith/c
                  index/c
                  log/c
                  reduce-or-variadic/c
                  tensor-or-real/c
                  unary-numeric/c)
         (only-in "error.rkt" check-handle)
         (only-in "ops.rkt" tensor-shape)
         (only-in "raw/elementwise.rkt"
                  tr-add-scalar/raw
                  tr-add/raw
                  tr-div-scalar/raw
                  tr-div/raw
                  tr-exp/raw
                  tr-gelu/raw
                  tr-log/raw
                  tr-mul-scalar/raw
                  tr-mul/raw
                  tr-neg/raw
                  tr-pow-scalar/raw
                  tr-pow/raw
                  tr-relu/raw
                  tr-sigmoid/raw
                  tr-sqrt/raw
                  tr-sub-scalar/raw
                  tr-sub/raw
                  tr-tanh/raw)
         (only-in "raw/linalg.rkt"
                  tr-dot/raw
                  tr-matmul/raw
                  tr-mm/raw
                  tr-mv/raw)
         (only-in "raw/reduce.rkt"
                  tr-argmax-all/raw
                  tr-argmax/raw
                  tr-log-softmax/raw
                  tr-max/raw
                  tr-mean/raw
                  tr-min/raw
                  tr-softmax/raw
                  tr-sum/raw)
         (only-in "raw/shape-ops.rkt"
                  tr-cat/raw
                  tr-permute/raw
                  tr-reshape/raw
                  tr-squeeze-dim/raw
                  tr-squeeze/raw
                  tr-stack/raw
                  tr-transpose/raw
                  tr-unsqueeze/raw
                  tr-view/raw)
         (only-in "structs.rkt" tensor? wrap-tensor))

(define (wrap who h)
  (wrap-tensor (check-handle who h)))

;; --------------------------------------------------------------- shape ops

(define/checked-out (reshape t . dims)
  (-> tensor? index/c ... tensor?)
  (wrap 'reshape (tr-reshape/raw t (list->s64vector dims) (length dims))))

(define/contract-out (view t . dims)
  (-> tensor? index/c ... tensor?)
  (wrap 'view (tr-view/raw t (list->s64vector dims) (length dims))))

(define/contract-out (transpose t dim0 dim1)
  (-> tensor? index/c index/c tensor?)
  (wrap 'transpose (tr-transpose/raw t dim0 dim1)))

(define/contract-out (permute t . dims)
  (-> tensor? index/c ... tensor?)
  (wrap 'permute (tr-permute/raw t (list->s64vector dims) (length dims))))

(define/contract-out (T x) ;; noqa
  (-> tensor? tensor?)
  (apply permute x (reverse (build-list (length (tensor-shape x)) values))))

(define/contract-out (squeeze t [dim #f])
  (->* [tensor?] [index/c] tensor?)
  (if dim
      (wrap 'squeeze (tr-squeeze-dim/raw t dim))
      (wrap 'squeeze (tr-squeeze/raw t))))

(define/checked-out (unsqueeze t dim)
  (-> tensor? index/c tensor?)
  (wrap 'unsqueeze (tr-unsqueeze/raw t dim)))

(define/contract-out (cat ts [dim 0])
  (->* [(non-empty-listof tensor?)] [index/c] tensor?)
  (wrap 'cat (tr-cat/raw ts (length ts) dim)))

(define/contract-out (stack ts [dim 0])
  (->* [(non-empty-listof tensor?)] [index/c] tensor?)
  (wrap 'stack (tr-stack/raw ts (length ts) dim)))

;; -------------------------------------------------------------- elementwise

(define (binary-dispatch who t-op s-op a b swapped-scalar)
  (cond
    [(and (tensor? a) (tensor? b)) (wrap who (t-op a b))]
    [(and (tensor? a) (real? b)) (wrap who (s-op a (exact->inexact b)))]
    [(and (real? a) (tensor? b)) (swapped-scalar (exact->inexact a) b)]
    [else (error who "expected at least one tensor, got ~e and ~e" a b)]))

(define/checked-out (add a b)
  binary-arith/c
  (binary-dispatch 'add tr-add/raw tr-add-scalar/raw a b
                   (lambda (s t) (add t s))))

(define/checked-out (sub a b)
  binary-arith/c
  (binary-dispatch 'sub tr-sub/raw tr-sub-scalar/raw a b
                   (lambda (s t) (add (neg t) s))))

(define/checked-out (mul a b)
  binary-arith/c
  (binary-dispatch 'mul tr-mul/raw tr-mul-scalar/raw a b
                   (lambda (s t) (mul t s))))

(define/checked-out (div a b)
  binary-arith/c
  (binary-dispatch 'div tr-div/raw tr-div-scalar/raw a b
                   (lambda (s t) (mul (pow t -1) s))))

(define/contract-out (pow a b)
  (-> tensor? tensor-or-real/c tensor?)
  (cond
    [(and (tensor? a) (tensor? b)) (wrap 'pow (tr-pow/raw a b))]
    [(and (tensor? a) (real? b))
     (wrap 'pow (tr-pow-scalar/raw a (exact->inexact b)))]
    [else (error 'pow "expected a tensor base, got ~e and ~e" a b)]))

(define/checked-out (neg t)
  (-> tensor? tensor?)
  (wrap 'neg (tr-neg/raw t)))

(define/contract-out (exp v)
  unary-numeric/c
  (if (tensor? v) (wrap 'exp (tr-exp/raw v)) (base:exp v)))

(define/contract-out (log v [base #f])
  log/c
  (cond
    [(and (tensor? v) base) (error 'log "tensor log takes no base argument")]
    [(tensor? v) (wrap 'log (tr-log/raw v))]
    [base (base:log v base)]
    [else (base:log v)]))

(define/contract-out (sqrt v)
  unary-numeric/c
  (if (tensor? v) (wrap 'sqrt (tr-sqrt/raw v)) (base:sqrt v)))

(define/contract-out (relu t)
  (-> tensor? tensor?)
  (wrap 'relu (tr-relu/raw t)))

(define/contract-out (sigmoid t)
  (-> tensor? tensor?)
  (wrap 'sigmoid (tr-sigmoid/raw t)))

(define/contract-out (tanh v)
  unary-numeric/c
  (if (tensor? v) (wrap 'tanh (tr-tanh/raw v)) (base:tanh v)))

(define/contract-out (gelu t)
  (-> tensor? tensor?)
  (wrap 'gelu (tr-gelu/raw t)))

;; -------------------------------------------------------------- reductions

(define/checked-out (sum t)
  (-> tensor? tensor?)
  (wrap 'sum (tr-sum/raw t)))

(define/contract-out (mean t)
  (-> tensor? tensor?)
  (wrap 'mean (tr-mean/raw t)))

(define/contract-out (max v . rest)
  reduce-or-variadic/c
  (if (tensor? v)
      (if (null? rest)
          (wrap 'max (tr-max/raw v))
          (error 'max "tensor max takes a single tensor"))
      (apply base:max v rest)))

(define/contract-out (min v . rest)
  reduce-or-variadic/c
  (if (tensor? v)
      (if (null? rest)
          (wrap 'min (tr-min/raw v))
          (error 'min "tensor min takes a single tensor"))
      (apply base:min v rest)))

(define/contract-out (argmax t [dim #f] #:keepdim [keepdim #f])
  argmax/c
  (cond
    [(tensor? t)
     (if dim
         (wrap 'argmax (tr-argmax/raw t dim keepdim))
         (wrap 'argmax (tr-argmax-all/raw t)))]
    [(procedure? t) (base:argmax t dim)]
    [else (error 'argmax "expected a tensor or a procedure, got ~e" t)]))

(define/contract-out (softmax t dim)
  (-> tensor? index/c tensor?)
  (wrap 'softmax (tr-softmax/raw t dim)))

(define/contract-out (log-softmax t dim)
  (-> tensor? index/c tensor?)
  (wrap 'log-softmax (tr-log-softmax/raw t dim)))

;; ------------------------------------------------------------------ linalg

(define/checked-out (matmul a b)
  (-> tensor? tensor? tensor?)
  (wrap 'matmul (tr-matmul/raw a b)))

(define/contract-out (mm a b)
  (-> tensor? tensor? tensor?)
  (wrap 'mm (tr-mm/raw a b)))

(define/contract-out (mv a b)
  (-> tensor? tensor? tensor?)
  (wrap 'mv (tr-mv/raw a b)))

(define/contract-out (dot a b)
  (-> tensor? tensor? tensor?)
  (wrap 'dot (tr-dot/raw a b)))
