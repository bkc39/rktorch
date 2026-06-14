#lang racket/base

;; define-generated-op — the expansion target for torch/generated.rkt.
;;
;; The codegen generator used to emit fully-expanded Racket (a raw
;; define-torch shard plus a wrapper body per op); now it emits one
;; compact form per op and this macro owns the expansion, so the
;; Racket-side marshalling knowledge lives here (reviewable, hygienic)
;; instead of in Python string templates:
;;
;;   (define-generated-op matmul tr_gen_matmul
;;     ([self tensor] [other tensor]))
;;
;; expands into the raw FFI binding (GC-managed via the allocator wrap,
;; like every tensor-returning binding) and the public uncontracted
;; wrapper. Argument kinds mirror the generator IR / parity manifest:
;;
;;   tensor           _Tensor (the wrapper struct passes via prop:cpointer)
;;   optional-tensor  _Tensor/null (a tensor or #f -> handle or NULL)
;;   scalar           at::Scalar marshalled as _double
;;   double           _double
;;   int64            _int64
;;   bool             _stdbool
;;   int-array        list of exact integers -> (s64vector, length) pair
;;   tensor-list      list of tensors -> (tensor array, length) pair
;;
;; The #:inplace form (after c-id) marks an op that mutates its first
;; argument (the receiver) and returns it; its raw binding returns an int
;; status (0 ok) instead of a fresh handle, so it is not allocator-wrapped:
;;
;;   (define-generated-op add-tensor! tr_gen_add__tensor #:inplace
;;     ([self tensor] [other tensor] [alpha scalar]))

(require (for-syntax racket/base
                     racket/syntax
                     ;; whole-module on purpose: syntax-parse patterns
                     ;; reference many exported bindings
                     syntax/parse/pre)
         (only-in ffi/unsafe _fun _int _int32 _int64 _double _list _stdbool)
         (only-in ffi/unsafe/alloc allocator)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/syntax.rkt"
                  _Tensor
                  _Tensor/null
                  define-torch
                  tr-tensor-free/raw)
         (only-in "structs.rkt" wrap-tensor))

(provide define-generated-op)

;; optional-dtype marshals a ScalarType as its at::ScalarType code, with -1
;; for #f (c10::nullopt). v2 is float32-only, so #f is the live case; the
;; named dtypes mirror torch/foreign tr_dtype for when v3 widens this.
(define (opt-dtype->code d)
  (case d
    [(#f) -1]
    [(int64) 4]
    [(float32) 6]
    [(float64) 7]
    [else (error 'define-generated-op "unsupported optional dtype: ~e" d)]))

;; For one [arg kind] pair, the pieces of the expansion:
;;   _fun param specs (one, or more for the array/optional kinds' split
;;   into pointer/length/presence), and the raw-call argument expressions
;;   the wrapper passes. Optional value kinds take a value-or-#f and split
;;   into the value plus a presence flag (or a -1 sentinel for dtype), so
;;   the wrapper never passes a NULL pointer for a value type.
(define-for-syntax (kind-pieces stx arg kind)
  (define len-arg (format-id arg "~a-len" arg))
  (define has-arg (format-id arg "~a-has" arg))
  (case kind
    [(tensor) (values (list #`(#,arg : _Tensor)) (list arg))]
    [(optional-tensor)
     ;; _Tensor/null shares the 'Tensor tag, so the wrapper marshals like
     ;; _Tensor; #f marshals to NULL (== c10::nullopt on the C side).
     (values (list #`(#,arg : _Tensor/null)) (list arg))]
    [(scalar double) (values (list #`(#,arg : _double)) (list arg))]
    [(int64) (values (list #`(#,arg : _int64)) (list arg))]
    [(bool) (values (list #`(#,arg : _stdbool)) (list arg))]
    [(int-array)
     (values (list #`(#,arg : (_s64vector i)) #`(#,len-arg : _int64))
             (list #`(list->s64vector #,arg) #`(length #,arg)))]
    [(tensor-list)
     (values (list #`(#,arg : (_list i _Tensor)) #`(#,len-arg : _int64))
             (list arg #`(length #,arg)))]
    [(optional-int64)
     (values (list #`(#,arg : _int64) #`(#,has-arg : _stdbool))
             (list #`(if #,arg #,arg 0) #`(and #,arg #t)))]
    [(optional-int-array)
     (values (list #`(#,arg : (_s64vector i)) #`(#,len-arg : _int64)
                   #`(#,has-arg : _stdbool))
             (list #`(list->s64vector (or #,arg '()))
                   #`(if #,arg (length #,arg) 0)
                   #`(and #,arg #t)))]
    [(optional-dtype)
     (values (list #`(#,arg : _int32)) (list #`(opt-dtype->code #,arg)))]
    [else
     (raise-syntax-error 'define-generated-op
                         (format "unknown argument kind: ~a" kind)
                         stx)]))

;; Build the _fun param specs and the raw-call argument expressions for a
;; list of [arg kind] pairs (each kind contributes one spec, or two for the
;; array kinds' pointer+length split).
(define-for-syntax (build-pieces stx args kinds)
  (define-values (rev-specs rev-call-args)
    (for/fold ([specs '()] [call-args '()])
              ([a (in-list args)]
               [k (in-list kinds)])
      (define-values (s c) (kind-pieces stx a k))
      (values (append (reverse s) specs)
              (append (reverse c) call-args))))
  (values (reverse rev-specs) (reverse rev-call-args)))

(define-syntax (define-generated-op stx)
  (syntax-parse stx
    ;; In-place: the raw binding returns an int status; the wrapper checks it
    ;; and returns the (now-mutated) receiver, like torch.Tensor.add_.
    [(_ name:id c-id:id #:inplace ([arg:id kind:id] ...+))
     (define-values (specs call-args)
       (build-pieces stx
                     (syntax->list #'(arg ...))
                     (syntax->datum #'(kind ...))))
     (with-syntax ([raw-name (format-id #'name "~a/raw" #'name)]
                   [(spec ...) specs]
                   [(call-arg ...) call-args]
                   [recv (car (syntax->list #'(arg ...)))])
       #'(begin
           (define-torch raw-name
             (_fun spec ... -> _int)
             #:c-id c-id)
           ;; recv is the first formal, still live in this body, so returning
           ;; it after the C call is GC-safe (the _fun call also pins it for
           ;; the duration of the in-place mutation).
           (define (name arg ...)
             (check-ok (raw-name call-arg ...) 'name)
             recv)))]
    [(_ name:id c-id:id ([arg:id kind:id] ...))
     (define-values (specs call-args)
       (build-pieces stx
                     (syntax->list #'(arg ...))
                     (syntax->datum #'(kind ...))))
     (with-syntax ([raw-name (format-id #'name "~a/raw" #'name)]
                   [(spec ...) specs]
                   [(call-arg ...) call-args])
       #'(begin
           (define-torch raw-name
             (_fun spec ... -> _Tensor/null)
             #:c-id c-id
             #:wrap (allocator tr-tensor-free/raw))
           (define (name arg ...)
             (wrap-tensor
              (check-handle 'name (raw-name call-arg ...))))))]))
