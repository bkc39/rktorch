#lang racket/base

;; define-generated-op — the expansion target for torch/generated.rkt.
;; Each op form expands into the raw FFI binding (allocator-wrapped when
;; tensor-returning) plus the public uncontracted wrapper, so the
;; Racket-side marshalling knowledge lives here rather than in the Python
;; codegen templates. Argument kinds mirror the generator IR:
;;
;;   tensor           _Tensor (the wrapper struct passes via prop:cpointer)
;;   optional-tensor  _Tensor/null (a tensor or #f -> handle or NULL)
;;   scalar           at::Scalar marshalled as _double
;;   double / int64 / bool    _double / _int64 / _stdbool
;;   int-array        list of exact integers -> (s64vector, length) pair
;;   tensor-list      list of tensors -> (tensor array, length) pair
;;
;; #:inplace marks an op that mutates its first argument and returns it;
;; its raw binding returns an int status, so it is not allocator-wrapped.

(require (for-syntax racket/base
                     racket/syntax
                     ;; whole-module on purpose: syntax-parse patterns
                     ;; reference many exported bindings
                     syntax/parse/pre)
         (only-in ffi/unsafe _fun _int _int32 _int64 _double _list _stdbool)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/memory.rkt" tensor-allocator tensor-allocator/rng)
         (only-in "raw/syntax.rkt" _Tensor _Tensor/null define-torch)
         (only-in "structs.rkt" wrap-tensor))

(provide define-generated-op)

;; optional-dtype marshals a ScalarType as its at::ScalarType code, with
;; -1 for #f (c10::nullopt); codes mirror tr_dtype.
(define (opt-dtype->code d)
  (case d
    [(#f) -1]
    [(int64) 4]
    [(float32) 6]
    [(float64) 7]
    [else (error 'define-generated-op "unsupported optional dtype: ~e" d)]))

;; For one [arg kind] pair: the _fun param specs (array/optional kinds
;; split into pointer/length/presence) and the raw-call argument
;; expressions. Optional value kinds pass a presence flag or sentinel, so
;; the wrapper never passes a NULL pointer for a value type.
(define-for-syntax (kind-pieces stx arg kind)
  (define len-arg (format-id arg "~a-len" arg))
  (define has-arg (format-id arg "~a-has" arg))
  (case kind
    [(tensor) (values (list #`(#,arg : _Tensor)) (list arg))]
    [(optional-tensor)
     ;; #f marshals to NULL (== c10::nullopt on the C side)
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
     ;; (or arg 0): the value when present, else a don't-care 0 paired with
     ;; has=#f. (if arg arg 0) reads as if 0 meant "absent" — it doesn't.
     (values (list #`(#,arg : _int64) #`(#,has-arg : _stdbool))
             (list #`(or #,arg 0) #`(and #,arg #t)))]
    [(optional-int-array)
     ;; #f or '() is absent; pair? gates the has flag so an empty list never
     ;; marshals as a present-but-empty dim (which is ambiguous).
     (values (list #`(#,arg : (_s64vector i)) #`(#,len-arg : _int64)
                   #`(#,has-arg : _stdbool))
             (list #`(list->s64vector (or #,arg '()))
                   #`(if (pair? #,arg) (length #,arg) 0)
                   #`(and (pair? #,arg) #t)))]
    [(optional-dtype)
     (values (list #`(#,arg : _int32)) (list #`(opt-dtype->code #,arg)))]
    [else
     (raise-syntax-error 'define-generated-op
                         (format "unknown argument kind: ~a" kind)
                         stx)]))

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
           ;; recv stays live in this body, so returning it after the C
           ;; call is GC-safe (_fun pins it for the mutation).
           (define (name arg ...)
             (check-ok (raw-name call-arg ...) 'name)
             recv)))]
    ;; RNG ops get the no-retry allocator wrap — a collect-and-retry
    ;; would draw from the generator twice and break seeded parity.
    [(_ name:id c-id:id #:rng ([arg:id kind:id] ...))
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
             #:wrap tensor-allocator/rng)
           (define (name arg ...)
             (wrap-tensor
              (check-handle 'name (raw-name call-arg ...))))))]
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
             #:wrap tensor-allocator)
           (define (name arg ...)
             (wrap-tensor
              (check-handle 'name (raw-name call-arg ...))))))]))
