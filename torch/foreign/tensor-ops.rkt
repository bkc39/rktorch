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
         (only-in racket/contract/base
                  -> ->* ->i list/c non-empty-listof or/c unsupplied-arg?)
         (only-in racket/list append-map [argmax base:argmax])
         (only-in racket/math [tanh base:tanh])
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "autograd-ops.rkt" requires-grad!)
         (only-in "contracts.rkt"
                  argmax/c binary-arith/c index/c log/c
                  reduce-or-variadic/c tensor-or-real/c unary-numeric/c)
         (only-in "device-type.rkt" device/c)
         (only-in "error.rkt" check-handle)
         (only-in "ops.rkt"
                  device->type+index dims-rest/c dtype/c tensor-device
                  tensor-dtype tensor-shape)
         (only-in "raw/creation.rkt"
                  tr-arange-on/raw
                  tr-eye-on/raw
                  tr-from-data-i64-on-device/raw
                  tr-from-data-i64/raw
                  tr-from-data-on-device/raw
                  tr-from-data/raw
                  tr-full-on/raw
                  tr-ones-on/raw
                  tr-zeros-on/raw)
         (only-in "raw/random.rkt" tr-rand-on/raw tr-randn-on/raw)
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

(define shape-rest/c (or/c (list/c dims-rest/c) dims-rest/c))

(define (shape-of dims)
  (if (and (pair? dims) (list? (car dims))) (car dims) dims))

;; the device and dtype go into native construction — never a default-device
;; scope or a construct-then-move hop through another device
(define (placement device dtype)
  (define-values (type index)
    (if device (device->type+index device) (values 'keep 0)))
  (values type index (or dtype 'keep)))

(define (finish out requires-grad?)
  (if requires-grad? (requires-grad! out) out))

(define float-dtype/c (or/c 'float32 'float64))

(define (shaped who raw dims device dtype requires-grad? . extra)
  (define shape (shape-of dims))
  (define-values (type index dt) (placement device dtype))
  (finish (wrap who
                (apply raw (list->s64vector shape) (length shape)
                       (append extra (list type index dt))))
          requires-grad?))

(define/contract-out (zeros #:device [device #f] #:dtype [dtype #f]
                            #:requires-grad? [requires-grad? #f]
                            . dims)
  (->* [] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'zeros tr-zeros-on/raw dims device dtype requires-grad?))

(define/contract-out (ones #:device [device #f] #:dtype [dtype #f]
                           #:requires-grad? [requires-grad? #f]
                           . dims)
  (->* [] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'ones tr-ones-on/raw dims device dtype requires-grad?))

;; the fill crosses the FFI as a double: an int64 fill outside the exact
;; range of a double would round silently, so the contract refuses it
(define (fill-crosses-exactly? value dtype)
  (or (unsupplied-arg? dtype)
      (not (eq? dtype 'int64))
      (not (exact-integer? value))
      (= (exact->inexact value) value)
      "an int64 fill value must be exactly representable as a double"))

(define/contract-out (full value #:device [device #f] #:dtype [dtype #f]
                           #:requires-grad? [requires-grad? #f]
                           . dims)
  (->i ([value real?])
       (#:device [device device/c]
        #:dtype [dtype dtype/c]
        #:requires-grad? [requires-grad? boolean?])
       #:rest [dims shape-rest/c]
       #:pre/desc (value dtype) (fill-crosses-exactly? value dtype)
       [result tensor?])
  (shaped 'full tr-full-on/raw dims device dtype requires-grad?
          (exact->inexact value)))

(define/contract-out (randn #:device [device #f] #:dtype [dtype #f]
                            #:requires-grad? [requires-grad? #f]
                            . dims)
  (->* [] [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'randn tr-randn-on/raw dims device dtype requires-grad?))

(define/contract-out (rand #:device [device #f] #:dtype [dtype #f]
                           #:requires-grad? [requires-grad? #f]
                           . dims)
  (->* [] [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'rand tr-rand-on/raw dims device dtype requires-grad?))

(define (like t device dtype)
  (values (or device (tensor-device t)) (or dtype (tensor-dtype t))))

(define/contract-out (zeros-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                 #:requires-grad? [requires-grad? #f])
  (->* [tensor?] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (zeros (tensor-shape t) #:device dev #:dtype dt
         #:requires-grad? requires-grad?))

(define/contract-out (ones-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                #:requires-grad? [requires-grad? #f])
  (->* [tensor?] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (ones (tensor-shape t) #:device dev #:dtype dt
        #:requires-grad? requires-grad?))

(define/contract-out (full-like t value #:device [device #f] #:dtype [dtype #f] ;; noqa
                                #:requires-grad? [requires-grad? #f])
  (->* [tensor? real?]
       [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (full value (tensor-shape t) #:device dev #:dtype dt
        #:requires-grad? requires-grad?))

(define/contract-out (randn-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                 #:requires-grad? [requires-grad? #f])
  (->* [tensor?]
       [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (randn (tensor-shape t) #:device dev #:dtype dt
         #:requires-grad? requires-grad?))

(define/contract-out (rand-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                #:requires-grad? [requires-grad? #f])
  (->* [tensor?]
       [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (rand (tensor-shape t) #:device dev #:dtype dt
        #:requires-grad? requires-grad?))

(define/contract-out (arange a [b #f] [c #f]
                             #:device [device #f] #:dtype [dtype #f]
                             #:requires-grad? [requires-grad? #f])
  (->* [real?]
       [real? real?
        #:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (start end step)
    (cond
      [(not b) (values 0 a 1)]
      [(not c) (values a b 1)]
      [else (values a b c)]))
  (define-values (type index dt) (placement device dtype))
  (finish (wrap 'arange
                (tr-arange-on/raw (exact->inexact start)
                                  (exact->inexact end)
                                  (exact->inexact step)
                                  type index dt))
          requires-grad?))

(define/contract-out (eye n [m n]
                          #:device [device #f] #:dtype [dtype #f]
                          #:requires-grad? [requires-grad? #f])
  (->* [exact-nonnegative-integer?]
       [exact-nonnegative-integer?
        #:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (type index dt) (placement device dtype))
  (finish (wrap 'eye (tr-eye-on/raw n m type index dt)) requires-grad?))

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
