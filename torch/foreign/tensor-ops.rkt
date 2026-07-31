#lang racket/base

;; Safe wrappers for the v1 op tranche (creation, shape, elementwise,
;; reductions, linalg), built on the raw layer + the tensor wrapper struct.
;; Contracts live in ../foreign.rkt.
;;
;; Naming: ops whose names collide with racket/base or racket/list
;; (exp log sqrt tanh max min argmax) are generic — given a tensor they hit
;; libtorch, given numbers they defer to the original — so
;; `(require torch)` doesn't break numeric code.
;; Binary arithmetic accepts a real on either side via the *_scalar shims.

(require (only-in ffi/vector list->f32vector list->s64vector)
         (only-in racket/base
                  [exp base:exp]
                  [log base:log]
                  [max base:max]
                  [min base:min]
                  [sqrt base:sqrt])
         ;; tanh is not in racket/base, but the full `racket` language
         ;; re-exports it from racket/math — so the dispatch shim still
         ;; matters for #lang racket users.
         (only-in racket/math [tanh base:tanh])
         (only-in racket/list [argmax base:argmax] flatten)
         (only-in "error.rkt" check-handle)
         (only-in "raw/creation.rkt"
                  tr-arange/raw
                  tr-eye/raw
                  tr-from-data/raw
                  tr-full/raw
                  tr-ones/raw
                  tr-zeros/raw)
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
         (only-in "raw/linalg.rkt" tr-dot/raw tr-matmul/raw tr-mm/raw tr-mv/raw)
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
         (only-in "autograd-ops.rkt" requires-grad!)
         (only-in "structs.rkt" tensor? wrap-tensor))

(provide zeros
         ones
         full
         arange
         eye
         tensor
         reshape
         view
         transpose
         permute
         squeeze
         unsqueeze
         cat
         stack
         add
         sub
         mul
         div
         pow
         neg
         exp
         log
         sqrt
         relu
         sigmoid
         tanh
         gelu
         sum
         mean
         max
         min
         argmax
         softmax
         log-softmax
         matmul
         mm
         mv
         dot)

(define (wrap who h)
  (wrap-tensor (check-handle who h)))

;; ---------------------------------------------------------------- creation

(define (zeros . dims)
  (wrap 'zeros (tr-zeros/raw (list->s64vector dims) (length dims))))

(define (ones . dims)
  (wrap 'ones (tr-ones/raw (list->s64vector dims) (length dims))))

;; (full value dim ...) — argument order flipped from torch.full(shape, value)
;; so the shape can stay variadic like zeros/ones/randn.
(define (full value . dims)
  (wrap 'full
        (tr-full/raw (list->s64vector dims)
                     (length dims)
                     (exact->inexact value))))

(define arange
  (case-lambda
    [(end) (arange 0 end 1)]
    [(start end) (arange start end 1)]
    [(start end step)
     (wrap 'arange
           (tr-arange/raw (exact->inexact start)
                          (exact->inexact end)
                          (exact->inexact step)))]))

(define (eye n [m n])
  (wrap 'eye (tr-eye/raw n m)))

;; Shape inference walks the first element of each nesting level, exactly like
;; torch.tensor; the numel check below (and again in C) rejects ragged input.
(define (nested-dims data)
  (cond
    [(not (list? data)) '()]
    [(null? data) '(0)]
    [else (cons (length data) (nested-dims (car data)))]))

(define (tensor data #:requires-grad? [requires-grad? #f])
  (define dims (nested-dims data))
  (define flat (if (list? data) (flatten data) (list data)))
  (unless (= (length flat) (apply * dims))
    (error 'tensor "ragged nested list; dims ~a need ~a values, got ~a"
           dims (apply * dims) (length flat)))
  (define out
    (wrap 'tensor
          (tr-from-data/raw (list->f32vector (map exact->inexact flat))
                            (length flat)
                            (list->s64vector dims)
                            (length dims))))
  (if requires-grad? (requires-grad! out) out))

;; --------------------------------------------------------------- shape ops

(define (reshape t . dims)
  (wrap 'reshape (tr-reshape/raw t (list->s64vector dims) (length dims))))

(define (view t . dims)
  (wrap 'view (tr-view/raw t (list->s64vector dims) (length dims))))

(define (transpose t dim0 dim1)
  (wrap 'transpose (tr-transpose/raw t dim0 dim1)))

(define (permute t . dims)
  (wrap 'permute (tr-permute/raw t (list->s64vector dims) (length dims))))

(define (squeeze t [dim #f])
  (if dim
      (wrap 'squeeze (tr-squeeze-dim/raw t dim))
      (wrap 'squeeze (tr-squeeze/raw t))))

(define (unsqueeze t dim)
  (wrap 'unsqueeze (tr-unsqueeze/raw t dim)))

(define (cat ts [dim 0])
  (wrap 'cat (tr-cat/raw ts (length ts) dim)))

(define (stack ts [dim 0])
  (wrap 'stack (tr-stack/raw ts (length ts) dim)))

;; -------------------------------------------------------------- elementwise

;; Dispatch one binary op over tensor/real argument combinations.
;; swapped-scalar handles (op real tensor) and must be the algebraic rewrite
;; of putting the scalar on the left.
(define (binary-dispatch who t-op s-op a b swapped-scalar)
  (cond
    [(and (tensor? a) (tensor? b)) (wrap who (t-op a b))]
    [(and (tensor? a) (real? b)) (wrap who (s-op a (exact->inexact b)))]
    [(and (real? a) (tensor? b)) (swapped-scalar (exact->inexact a) b)]
    [else (error who "expected at least one tensor, got ~e and ~e" a b)]))

(define (add a b)
  (binary-dispatch 'add tr-add/raw tr-add-scalar/raw a b
                   (lambda (s t) (add t s))))

(define (sub a b)
  (binary-dispatch 'sub tr-sub/raw tr-sub-scalar/raw a b
                   (lambda (s t) (add (neg t) s))))

(define (mul a b)
  (binary-dispatch 'mul tr-mul/raw tr-mul-scalar/raw a b
                   (lambda (s t) (mul t s))))

(define (div a b)
  (binary-dispatch 'div tr-div/raw tr-div-scalar/raw a b
                   (lambda (s t) (mul (pow t -1) s))))

(define (pow a b)
  (cond
    [(and (tensor? a) (tensor? b)) (wrap 'pow (tr-pow/raw a b))]
    [(and (tensor? a) (real? b))
     (wrap 'pow (tr-pow-scalar/raw a (exact->inexact b)))]
    [else (error 'pow "expected a tensor base, got ~e and ~e" a b)]))

(define (neg t)
  (wrap 'neg (tr-neg/raw t)))

(define (exp v)
  (if (tensor? v) (wrap 'exp (tr-exp/raw v)) (base:exp v)))

(define (log v [base #f])
  (cond
    [(and (tensor? v) base) (error 'log "tensor log takes no base argument")]
    [(tensor? v) (wrap 'log (tr-log/raw v))]
    [base (base:log v base)]
    [else (base:log v)]))

(define (sqrt v)
  (if (tensor? v) (wrap 'sqrt (tr-sqrt/raw v)) (base:sqrt v)))

(define (relu t)
  (wrap 'relu (tr-relu/raw t)))

(define (sigmoid t)
  (wrap 'sigmoid (tr-sigmoid/raw t)))

(define (tanh v)
  (if (tensor? v) (wrap 'tanh (tr-tanh/raw v)) (base:tanh v)))

;; Exact (erf-based) gelu, approximate='none' — the transformer MLP
;; activation. No racket/base collision, so no shadow dispatch.
(define (gelu t)
  (wrap 'gelu (tr-gelu/raw t)))

;; -------------------------------------------------------------- reductions

(define (sum t)
  (wrap 'sum (tr-sum/raw t)))

(define (mean t)
  (wrap 'mean (tr-mean/raw t)))

(define (max v . rest)
  (if (tensor? v)
      (if (null? rest)
          (wrap 'max (tr-max/raw v))
          (error 'max "tensor max takes a single tensor"))
      (apply base:max v rest)))

(define (min v . rest)
  (if (tensor? v)
      (if (null? rest)
          (wrap 'min (tr-min/raw v))
          (error 'min "tensor min takes a single tensor"))
      (apply base:min v rest)))

;; Shadows racket/list's argmax the same way exp/log shadow racket/base:
;; (argmax proc lst) still reaches the list version.
(define (argmax t [dim #f] #:keepdim [keepdim #f])
  (cond
    [(tensor? t)
     (if dim
         (wrap 'argmax (tr-argmax/raw t dim keepdim))
         (wrap 'argmax (tr-argmax-all/raw t)))]
    [(procedure? t) (base:argmax t dim)]
    [else (error 'argmax "expected a tensor or a procedure, got ~e" t)]))

(define (softmax t dim)
  (wrap 'softmax (tr-softmax/raw t dim)))

(define (log-softmax t dim)
  (wrap 'log-softmax (tr-log-softmax/raw t dim)))

;; ------------------------------------------------------------------ linalg

(define (matmul a b)
  (wrap 'matmul (tr-matmul/raw a b)))

(define (mm a b)
  (wrap 'mm (tr-mm/raw a b)))

(define (mv a b)
  (wrap 'mv (tr-mv/raw a b)))

(define (dot a b)
  (wrap 'dot (tr-dot/raw a b)))
