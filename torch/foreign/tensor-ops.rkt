#lang racket/base

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
         (only-in racket/contract/base -> ->* non-empty-listof or/c)
         (only-in racket/list append-map [argmax base:argmax])
         (only-in racket/math [tanh base:tanh])
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "autograd-ops.rkt" requires-grad!)
         (only-in "contracts.rkt"
                  arange/c argmax/c binary-arith/c index/c log/c
                  reduce-or-variadic/c tensor-or-real/c unary-numeric/c)
         (only-in "device-type.rkt" device/c)
         (only-in "error.rkt" check-handle)
         (only-in "ops.rkt" device->type+index dims-rest/c)
         (only-in "raw/creation.rkt"
                  tr-arange/raw
                  tr-eye/raw
                  tr-from-data-i64-on-device/raw
                  tr-from-data-i64/raw
                  tr-from-data-on-device/raw
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
         (only-in "structs.rkt" tensor? wrap-tensor))

(define (wrap who h)
  (wrap-tensor (check-handle who h)))

;; ---------------------------------------------------------------- creation

(define/contract-out (zeros . dims)
  (->* [] #:rest dims-rest/c tensor?)
  (wrap 'zeros (tr-zeros/raw (list->s64vector dims) (length dims))))

(define/contract-out (ones . dims)
  (->* [] #:rest dims-rest/c tensor?)
  (wrap 'ones (tr-ones/raw (list->s64vector dims) (length dims))))

(define/contract-out (full value . dims)
  (->* [real?] #:rest dims-rest/c tensor?)
  (wrap 'full
        (tr-full/raw (list->s64vector dims)
                     (length dims)
                     (exact->inexact value))))

(define/contract-out arange
  arange/c
  (case-lambda
    [(end) (arange 0 end 1)]
    [(start end) (arange start end 1)]
    [(start end step)
     (wrap 'arange
           (tr-arange/raw (exact->inexact start)
                          (exact->inexact end)
                          (exact->inexact step)))]))

(define/contract-out (eye n [m n])
  (->* [exact-nonnegative-integer?] [exact-nonnegative-integer?] tensor?)
  (wrap 'eye (tr-eye/raw n m)))

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
     (error 'tensor "cannot convert non-finite value to int64: ~e" x)]))

(define (infer-dtype flat)
  (cond
    [(null? flat) 'float32]
    [(andmap exact-integer? flat) 'int64]
    [else 'float32]))

(define/checked-out (tensor data
                            #:requires-grad? [requires-grad? #f]
                            #:device [device #f]
                            #:dtype [dtype #f])
  (->* [(or/c real? list? vector? f32vector? s64vector?)]
       [#:requires-grad? boolean?
        #:device (or/c #f device/c)
        #:dtype (or/c #f 'float32 'int64)]
       tensor?)
  (unless (memq dtype '(#f float32 int64))
    (error 'tensor "unsupported #:dtype (float32 or int64): ~e" dtype))
  (define dims (nested-dims data))
  (define-values (chosen payload numel)
    (cond
      [(and (f32vector? data) (not (eq? dtype 'int64)))
       (values 'float32 data (f32vector-length data))]
      [(and (s64vector? data) (not (eq? dtype 'float32)))
       (values 'int64 data (s64vector-length data))]
      [else
       (check-regular data dims 0)
       (define flat (sequence-flatten data))
       (case (or dtype (infer-dtype flat))
         [(int64)
          (values 'int64
                  (list->s64vector (map exact-int64 flat))
                  (length flat))]
         [else
          (values 'float32
                  (list->f32vector (map exact->inexact flat))
                  (length flat))])]))
  (define dim-vec (list->s64vector dims))
  (define-values (type index)
    (if device (device->type+index device) (values #f #f)))
  (define out
    (wrap 'tensor
          (case chosen
            [(int64)
             (if device
                 (tr-from-data-i64-on-device/raw payload numel
                                          dim-vec (length dims)
                                          type index)
                 (tr-from-data-i64/raw payload numel
                                       dim-vec (length dims)))]
            [else
             (if device
                 (tr-from-data-on-device/raw payload numel
                                      dim-vec (length dims)
                                      type index)
                 (tr-from-data/raw payload numel
                                   dim-vec (length dims)))])))
  (if requires-grad? (requires-grad! out) out))

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
