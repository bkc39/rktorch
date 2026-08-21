#lang racket/base

;; The `tensor` wrapper struct over the raw _Tensor cpointer.
;;
;; The cpointer is field 0, so `prop:cpointer 0` makes the struct itself stand
;; in for the handle — it threads straight through the raw layer's `_Tensor`
;; arguments with no manual unwrapping.  The struct also caches the shape (a
;; list of dimension sizes) queried once at wrap time, so `tensor-shape` and the
;; custom printer need no C round-trip.
;;
;; Lifetime: the raw constructor's `#:wrap tensor-allocator` auto-registers a
;; finalizer that calls `tr-tensor-free/finalizer` (the guarded finalizer-context
;; entry) and charges the #37 memory-pressure ledger.  The explicit
;; `tensor-free!` calls `tr-tensor-free/checked` — the
;; raising, `(deallocator)`-wrapped binding, so failures surface to the
;; deliberate caller and the pending finalizer is genuinely canceled — and
;; flips the cpointer tag, so a second free raises `exn:fail:contract` at
;; the contract boundary instead of double-freeing at the C level.

(require (only-in ffi/unsafe cpointer-has-tag? prop:cpointer set-cpointer-tag!)
         (only-in ffi/vector
                  f32vector->list
                  make-f32vector
                  make-s64vector
                  s64vector->list
                  s64vector-ref)
         (only-in racket/string string-join)
         (only-in "error.rkt" check-ok)
         (only-in "format.rkt" needs-sci-notation? tensor->pytorch-repr)
         (only-in "raw/memory.rkt" tr-tensor-free/checked)
         (only-in "raw/syntax.rkt" Tensor?)
         (only-in "raw/tensor.rkt"
                  tr-tensor-copy-data-i64/raw
                  tr-tensor-copy-data/raw
                  tr-tensor-dtype/raw
                  tr-tensor-print/raw
                  tr-tensor-shape/raw))

(provide (struct-out tensor-impl)
         tensor?
         tensor-handle
         tensor-impl-shape
         handle->string
         handle->repr
         wrap-tensor
         tensor-free!)

(define (shape->string dims)
  (if (null? dims) "scalar" (string-join (map number->string dims) "x")))

;; Render a live handle via ATen's ostream operator (the size-then-fill probe).
;; This is the same text libtorch's own `std::cout << tensor` produces, e.g.
;;    1.5410 -0.2934
;;   -2.1788  0.5684
;;   [ CPUFloatType{2,2} ]
;; It backs `tensor->string` and is the fallback when the PyTorch-style repr
;; can't be reproduced (scientific notation).
(define (handle->string h)
  (define-values (rc0 len0) (tr-tensor-print/raw h 0 (make-bytes 0)))
  (cond
    [(zero? rc0) ""]
    [(= rc0 2)
     (define buf (make-bytes len0))
     (define-values (rc1 len1) (tr-tensor-print/raw h len0 buf))
     (check-ok rc1 'tensor->string)
     (bytes->string/utf-8 buf #f 0 len1)]
    [else (check-ok rc0 'tensor->string) ""]))

;; Copy a live handle's values out as a flat row-major list of reals.
(define (handle->floats h dims)
  (define numel (apply * dims))
  (define out (make-f32vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data/raw h numel out))
  (check-ok rc 'tensor->repr)
  (f32vector->list out))

;; Copy an int64 handle's values out as a flat row-major list of EXACT
;; integers (#44) — the float path would corrupt values beyond 2^24.
(define (handle->ints h dims)
  (define numel (apply * dims))
  (define out (make-s64vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data-i64/raw h numel out))
  (check-ok rc 'tensor->repr)
  (s64vector->list out))

;; The PyTorch REPL form: `tensor([[...]])`, reproduced from the data + shape.
;; int64 tensors render bare integers (torch.tensor([1, 2, 3]) parity);
;; everything else falls back to ATen's printer for values PyTorch would
;; show in scientific notation (which the Racket-side formatter doesn't
;; yet reproduce). A failing dtype query falls through to the float path
;; (printing must never raise — the custom-write handler depends on it).
(define (handle->repr h dims)
  (define-values (dtype-rc dtype) (tr-tensor-dtype/raw h))
  (cond
    ;; 2 = TR_DTYPE_INT64 (the raw binding returns the plain int so an
    ;; out-of-enum dtype on the error path can't poison the unmarshal)
    [(and (zero? dtype-rc) (eqv? dtype 2))
     (tensor->pytorch-repr (handle->ints h dims) dims #:exact-integers? #t)]
    [else
     (define floats (handle->floats h dims))
     (if (needs-sci-notation? floats)
         (handle->string h)
         (tensor->pytorch-repr floats dims))]))

;; The REPL/`print`/`write` form mirrors the Python REPL: show the tensor's
;; contents as `tensor(...)`, not an opaque handle.  A freed handle (tag
;; flipped) prints a marker instead of touching C; any rendering error falls
;; back to the compact shape form so printing can never raise.
(struct tensor-impl (handle shape)
  #:reflection-name 'tensor
  #:property prop:cpointer 0
  #:property prop:custom-write
  (lambda (t port _mode)
    (define h (tensor-impl-handle t))
    (cond
      [(not (cpointer-has-tag? h 'Tensor))
       (fprintf port "#<tensor (freed)>")]
      [else
       (with-handlers ([exn:fail?
                        (lambda (_e)
                          (fprintf port "#<tensor:~a>"
                                   (shape->string (tensor-impl-shape t))))])
         (write-string (handle->repr h (tensor-impl-shape t)) port))])))

(define (tensor? v)
  (and (tensor-impl? v) (Tensor? v)))

(define tensor-handle tensor-impl-handle)

;; Query a handle's shape via the size-then-fill probe.  ndim==0 (a scalar)
;; returns rc=0 from the zero-capacity probe; ndim>0 returns rc=2 with the
;; required ndim, which we then allocate and fill.
(define (handle-shape h)
  (define-values (rc0 ndim0) (tr-tensor-shape/raw h 0 (make-s64vector 0)))
  (cond
    [(zero? rc0) '()]
    [(= rc0 2)
     (define dims (make-s64vector ndim0))
     (define-values (rc1 ndim1) (tr-tensor-shape/raw h ndim0 dims))
     (check-ok rc1 'tensor-shape)
     (for/list ([i (in-range ndim1)])
       (s64vector-ref dims i))]
    [else (check-ok rc0 'tensor-shape) '()]))

(define (wrap-tensor h)
  (tensor-impl h (handle-shape h)))

;; The tag flips even if the checked free raises (dynamic-wind): once the
;; free has been ATTEMPTED, the (deallocator)-consumed finalizer backstop
;; can no longer be relied on, so a live-looking tag would invite
;; use-after-free and double-free attempts with no safety net. On the
;; (exceptional) raising path the native handle may leak instead — the
;; right trade, and the raise still reaches the caller.
(define (tensor-free! t)
  (define h (tensor-handle t))
  (when (cpointer-has-tag? h 'Tensor)
    (dynamic-wind
     void
     ;; Breaks are deferred across the release: a break landing between
     ;; the ledger unaccount and the C free would propagate with the tag
     ;; flip still guaranteed below — leaving a handle the finalizer can
     ;; no longer free (tag mismatch, swallowed) — i.e. a permanent
     ;; native leak. The free is short; the break is delivered right
     ;; after.
     (lambda () (parameterize-break #f (tr-tensor-free/checked h)))
     (lambda () (set-cpointer-tag! h 'Tensor-freed)))))
