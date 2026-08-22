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
                  f64vector->list
                  make-f32vector
                  make-f64vector
                  make-s64vector
                  s64vector->list
                  s64vector-ref)
         (only-in racket/string string-join)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "format.rkt"
                  tensor->pytorch-repr
                  tensor-tree->pytorch-repr)
         (only-in "raw/memory.rkt" tr-tensor-free/checked)
         (only-in "raw/syntax.rkt" Tensor?)
         (only-in "raw/tensor.rkt"
                  dtype-code->symbol
                  tr-tensor-copy-data-f64/raw
                  tr-tensor-copy-data-i64/raw
                  tr-tensor-copy-data/raw
                  tr-tensor-dtype/raw
                  tr-tensor-narrow/raw
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

;; Copy a float64 handle's values out as doubles — full precision (the
;; float32 path truncates mantissas past 2^24).
(define (handle->doubles h dims)
  (define numel (apply * dims))
  (define out (make-f64vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data-f64/raw h numel out))
  (check-ok rc 'tensor->repr)
  (f64vector->list out))

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
;; torch's empty-tensor repr: "tensor([])" for the rank-1 float empty,
;; a size= clause whenever the shape isn't just (0), and a dtype suffix
;; exactly when the (absent) elements can't disambiguate int64.
(define (empty-repr dims dtype)
  (string-append
   "tensor([]"
   (if (equal? dims '(0))
       ""
       (format ", size=(~a)"
               (string-join (map number->string dims) ", ")))
   ;; torch appends the dtype exactly when the (absent) elements can't
   ;; disambiguate it from the float32 default
   (case dtype
     [(float64) ", dtype=torch.float64"]
     [(int64) ", dtype=torch.int64"]
     [(bool) ", dtype=torch.bool"]
     [else ""])
   ")"))

;; --- summarization (#45): PyTorch's edgeitems/threshold elision --------
;; A tensor with more than `threshold` elements prints only `edgeitems`
;; leading and trailing entries per oversized dimension, with 'ellipsis
;; markers where the middle was dropped. The edges are extracted with
;; native narrow slices, so the marshal-out cost scales with edgeitems,
;; never numel — a (zeros 1024 1024) repr copies 36 floats, not 2^20.
(define summarize-threshold 1000)
(define edgeitems 3)

;; Slice `len` entries starting at `start` along dimension `d` of `h`.
(define (slice h d start len)
  (check-handle 'tensor->repr (tr-tensor-narrow/raw h d start len)))

;; Build the summarized tree: recurse the leading dimensions (narrowing
;; per included index), marshal only leaf slices. `d` is the absolute
;; dimension index into `h`, which keeps its full rank (leading
;; length-1 dims) down the recursion; leaf copies flatten regardless.
(define (handle->summarized-tree h dims d leaf-values)
  (define n (car dims))
  (define last? (null? (cdr dims)))
  (define elide? (> n (* 2 edgeitems)))
  (cond
    [(and last? elide?)
     (append (leaf-values (slice h d 0 edgeitems) (list edgeitems))
             '(ellipsis)
             (leaf-values (slice h d (- n edgeitems) edgeitems)
                          (list edgeitems)))]
    [last? (leaf-values h (list n))]
    [else
     (define (child i)
       (handle->summarized-tree (slice h d i 1) (cdr dims) (add1 d)
                                leaf-values))
     (if elide?
         (append (for/list ([i (in-range edgeitems)]) (child i))
                 '(ellipsis)
                 (for/list ([i (in-range (- n edgeitems) n)]) (child i)))
         (for/list ([i (in-range n)]) (child i)))]))

(define (handle->repr h dims)
  (define-values (dtype-rc code) (tr-tensor-dtype/raw h))
  (define dtype (and (zero? dtype-rc) (dtype-code->symbol code)))
  (define numel (apply * dims))
  ;; summarization only engages when some dimension can actually elide:
  ;; above-threshold tensors whose dims are all <= 2*edgeitems print in
  ;; full (PyTorch behavior), through the direct path — no per-element
  ;; narrow slicing for a tree with nothing elided
  (define summarize?
    (and (> numel summarize-threshold)
         (for/or ([n (in-list dims)]) (> n (* 2 edgeitems)))))
  (cond
    [(zero? numel) (empty-repr dims dtype)]
    [summarize?
     (define leaf-values
       (case dtype
         [(int64) handle->ints]
         [(float64) handle->doubles]
         [else handle->floats]))
     (define tree (handle->summarized-tree h dims 0 leaf-values))
     (define mode
       (case dtype [(int64) 'exact-integers] [(bool) 'booleans] [else #f]))
     ;; the formatter itself picks sci notation from the SELECTED
     ;; values (PyTorch formats from the summarized population)
     (tensor-tree->pytorch-repr tree dims #:mode mode)]
    [(eq? dtype 'int64)
     (tensor->pytorch-repr (handle->ints h dims) dims
                           #:mode 'exact-integers)]
    [(eq? dtype 'bool)
     ;; the 0/1 mask arrives via the float copy path; the formatter
     ;; renders True/False (torch.tensor([True, False]) parity)
     (tensor->pytorch-repr (handle->floats h dims) dims
                           #:mode 'booleans)]
    [(eq? dtype 'float64)
     ;; full-precision doubles: sci mantissas past float32's 2^24 stay
     ;; right (the dtype suffix remains the documented TODO)
     (tensor->pytorch-repr (handle->doubles h dims) dims)]
    [else (tensor->pytorch-repr (handle->floats h dims) dims)]))

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
