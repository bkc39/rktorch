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

(require (only-in ffi/vector
                  f32vector->list
                  f32vector-length
                  f32vector?
                  list->f32vector
                  list->s64vector
                  s64vector->list
                  s64vector-length
                  s64vector?)
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
         (only-in racket/list append-map [argmax base:argmax])
         (only-in "error.rkt" check-handle)
         (only-in "ops.rkt" device->type+index)
         (only-in "raw/creation.rkt"
                  tr-arange/raw
                  tr-eye/raw
                  tr-from-data-i64-on/raw
                  tr-from-data-i64/raw
                  tr-from-data-on/raw
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
;; torch.tensor; check-regular then holds every sibling to these dims.
;; Any level may be a list, a vector, or (as a leaf run) a homogeneous
;; f32vector/s64vector — mixed nesting is fine, matching torch.tensor's
;; acceptance of any reasonable sequence nesting.
(define (nested-dims data)
  (cond
    [(list? data)
     (if (null? data) '(0) (cons (length data) (nested-dims (car data))))]
    [(vector? data)
     (if (zero? (vector-length data))
         '(0)
         (cons (vector-length data) (nested-dims (vector-ref data 0))))]
    [(f32vector? data) (list (f32vector-length data))]
    [(s64vector? data) (list (s64vector-length data))]
    [else '()]))

(define (sequence-children data)
  (cond
    [(list? data) data]
    [(vector? data) (vector->list data)]
    [(f32vector? data) (f32vector->list data)]
    [(s64vector? data) (s64vector->list data)]
    [else #f]))

(define (sequence-flatten data)
  (define kids (sequence-children data))
  (if kids (append-map sequence-flatten kids) (list data)))

;; Every sibling must match the dims inferred from the first element at
;; each level — a leaf count times out against (apply * dims) cannot
;; catch depth-ragged input like ((1 2) ((3) (4))), whose 4 leaves
;; satisfy 2x2 while the second row nests one level deeper.
;; torch.tensor validates recursively and raises here too.
(define (check-regular data dims d)
  (define kids (sequence-children data))
  (cond
    [(null? dims)
     (when kids
       (error 'tensor
              "ragged nested sequence: unexpected sequence at dim ~a: ~e"
              d data))]
    [(not kids)
     (error 'tensor
            (string-append "ragged nested sequence: expected sequence of"
                           " length ~a at dim ~a, got ~e")
            (car dims) d data)]
    [(not (= (length kids) (car dims)))
     (error 'tensor
            (string-append "ragged nested sequence: expected sequence of"
                           " length ~a at dim ~a, got length ~a")
            (car dims) d (length kids))]
    [else
     (for ([kid (in-list kids)])
       (check-regular kid (cdr dims) (add1 d)))]))

(define (exact-int64 x)
  (cond
    [(exact-integer? x) x]
    [(rational? x) (inexact->exact (truncate x))]
    [else
     ;; +inf.0/-inf.0/+nan.0 have no int64 value; raise tensor's own
     ;; error shape, not a raw conversion failure (torch errors here too)
     (error 'tensor "cannot convert non-finite value to int64: ~e" x)]))

(define (tensor data
                #:requires-grad? [requires-grad? #f]
                #:device [device #f]
                #:dtype [dtype #f])
  (unless (memq dtype '(#f float32 int64))
    (error 'tensor "unsupported #:dtype (float32 or int64): ~e" dtype))
  (define dims (nested-dims data))
  ;; dtype mirrors torch.tensor's inference (#44): all exact integers
  ;; infer int64; anything inexact (or an exact rational) infers
  ;; float32, PyTorch's default float dtype. #:dtype overrides either
  ;; way ('int64 truncates toward zero, torch's cast semantics).
  ;; Booleans and a float64 ingestion path are future work.
  ;; empty data stays float32 — torch.tensor([]) is float32, and the
  ;; vacuous andmap must not flip it to int64.
  ;; A top-level f32vector/s64vector already matching the target dtype
  ;; is handed to the FFI as-is — zero conversion copies (the C side
  ;; still clones, so there is no lifetime coupling to the buffer).
  (define-values (chosen payload numel)
    (cond
      [(and (f32vector? data) (not (eq? dtype 'int64)))
       (values 'float32 data (f32vector-length data))]
      [(and (s64vector? data) (not (eq? dtype 'float32)))
       (values 'int64 data (s64vector-length data))]
      [else
       (check-regular data dims 0)
       (define flat (sequence-flatten data))
       (define inferred
         (or dtype
             (if (and (pair? flat) (andmap exact-integer? flat))
                 'int64
                 'float32)))
       (case inferred
         [(int64)
          ;; exact integers marshal untouched — no float32 transit, so
          ;; values beyond 2^24 (and 2^53) stay exact
          (values 'int64
                  (list->s64vector (map exact-int64 flat))
                  (length flat))]
         [else
          (values 'float32
                  (list->f32vector (map exact->inexact flat))
                  (length flat))])]))
  ;; #:device passes the placement into NATIVE construction
  ;; (tr_from_data_on) rather than scoping the process-global default
  ;; (races concurrent constructors) or constructing-then-moving (the
  ;; construction itself routes host data through the default device, so
  ;; an explicitly-CPU tensor under a CUDA default would bounce
  ;; host->GPU->CPU — or CUDA-OOM). requires-grad! comes after placement
  ;; so the result is a LEAF on the target device, matching
  ;; torch.tensor(data, device=..., requires_grad=True).
  (define dim-vec (list->s64vector dims))
  (define-values (type index)
    (if device (device->type+index device) (values #f #f)))
  (define out
    (wrap 'tensor
          (case chosen
            [(int64)
             (if device
                 (tr-from-data-i64-on/raw payload numel
                                          dim-vec (length dims)
                                          type index)
                 (tr-from-data-i64/raw payload numel
                                       dim-vec (length dims)))]
            [else
             (if device
                 (tr-from-data-on/raw payload numel
                                      dim-vec (length dims)
                                      type index)
                 (tr-from-data/raw payload numel
                                   dim-vec (length dims)))])))
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
