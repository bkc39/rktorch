#lang racket/base

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

(define (handle->floats h dims)
  (define numel (apply * dims))
  (define out (make-f32vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data/raw h numel out))
  (check-ok rc 'tensor->repr)
  (f32vector->list out))

(define (handle->doubles h dims)
  (define numel (apply * dims))
  (define out (make-f64vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data-f64/raw h numel out))
  (check-ok rc 'tensor->repr)
  (f64vector->list out))

(define (handle->ints h dims)
  (define numel (apply * dims))
  (define out (make-s64vector numel))
  (define-values (rc _numel) (tr-tensor-copy-data-i64/raw h numel out))
  (check-ok rc 'tensor->repr)
  (s64vector->list out))

(define (empty-repr dims dtype)
  (string-append
   "tensor([]"
   (if (equal? dims '(0))
       ""
       (format ", size=(~a)"
               (string-join (map number->string dims) ", ")))
   (case dtype
     [(float64) ", dtype=torch.float64"]
     [(int64) ", dtype=torch.int64"]
     [(bool) ", dtype=torch.bool"]
     [else ""])
   ")"))

(define summarize-threshold 1000)
(define edgeitems 3)

;; The tag flips even when the free raises — an attempted free consumes the
;; finalizer backstop, so a live-looking tag would invite use-after-free.
;; Breaks stay deferred: one landing mid-free strands the handle forever.
(define (free-handle! h)
  (dynamic-wind
   void
   (lambda () (parameterize-break #f (tr-tensor-free/checked h)))
   (lambda () (set-cpointer-tag! h 'Tensor-freed))))

(define (call-with-slice h d start len proc)
  (define s (check-handle 'tensor->repr (tr-tensor-narrow/raw h d start len)))
  (begin0
    (proc s)
    (free-handle! s)))

(define (handle->summarized-tree h dims d leaf-values)
  (define n (car dims))
  (define last? (null? (cdr dims)))
  (define elide? (> n (* 2 edgeitems)))
  (cond
    [(and last? elide?)
     (append (call-with-slice h d 0 edgeitems
                              (lambda (s) (leaf-values s (list edgeitems))))
             '(ellipsis)
             (call-with-slice h d (- n edgeitems) edgeitems
                              (lambda (s) (leaf-values s (list edgeitems)))))]
    [last? (leaf-values h (list n))]
    [else
     (define (child i)
       (call-with-slice h d i 1
                        (lambda (s)
                          (handle->summarized-tree s (cdr dims) (add1 d)
                                                   leaf-values))))
     (if elide?
         (append (for/list ([i (in-range edgeitems)]) (child i))
                 '(ellipsis)
                 (for/list ([i (in-range (- n edgeitems) n)]) (child i)))
         (for/list ([i (in-range n)]) (child i)))]))

(define (handle->repr h dims)
  (define-values (dtype-rc code) (tr-tensor-dtype/raw h))
  (define dtype (and (zero? dtype-rc) (dtype-code->symbol code)))
  (define numel (apply * dims))
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
     (tensor-tree->pytorch-repr tree dims #:mode mode)]
    [(eq? dtype 'int64)
     (tensor->pytorch-repr (handle->ints h dims) dims
                           #:mode 'exact-integers)]
    [(eq? dtype 'bool)
     (tensor->pytorch-repr (handle->floats h dims) dims
                           #:mode 'booleans)]
    [(eq? dtype 'float64)
     (tensor->pytorch-repr (handle->doubles h dims) dims)]
    [else (tensor->pytorch-repr (handle->floats h dims) dims)]))

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

(define (tensor-free! t)
  (define h (tensor-handle t))
  (when (cpointer-has-tag? h 'Tensor)
    (free-handle! h)))
