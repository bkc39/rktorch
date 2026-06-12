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
;;   tensor       _Tensor (the wrapper struct passes via prop:cpointer)
;;   scalar       at::Scalar marshalled as _double
;;   double       _double
;;   int64        _int64
;;   bool         _stdbool
;;   int-array    list of exact integers -> (s64vector, length) pair
;;   tensor-list  list of tensors -> (tensor array, length) pair

(require (for-syntax racket/base
                     racket/syntax
                     ;; whole-module on purpose: syntax-parse patterns
                     ;; reference many exported bindings
                     syntax/parse/pre)
         (only-in ffi/unsafe _fun _int64 _double _list _stdbool)
         (only-in ffi/unsafe/alloc allocator)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in "error.rkt" check-handle)
         (only-in "raw/syntax.rkt"
                  _Tensor
                  _Tensor/null
                  define-torch
                  tr-tensor-free/raw)
         (only-in "structs.rkt" wrap-tensor))

(provide define-generated-op)

;; For one [arg kind] pair, the pieces of the expansion:
;;   _fun param specs (one, or two for the array kinds' pointer+length
;;   split), and the raw-call argument expressions the wrapper passes.
(define-for-syntax (kind-pieces stx arg kind)
  (define len-arg (format-id arg "~a-len" arg))
  (case kind
    [(tensor) (values (list #`(#,arg : _Tensor)) (list arg))]
    [(scalar double) (values (list #`(#,arg : _double)) (list arg))]
    [(int64) (values (list #`(#,arg : _int64)) (list arg))]
    [(bool) (values (list #`(#,arg : _stdbool)) (list arg))]
    [(int-array)
     (values (list #`(#,arg : (_s64vector i)) #`(#,len-arg : _int64))
             (list #`(list->s64vector #,arg) #`(length #,arg)))]
    [(tensor-list)
     (values (list #`(#,arg : (_list i _Tensor)) #`(#,len-arg : _int64))
             (list arg #`(length #,arg)))]
    [else
     (raise-syntax-error 'define-generated-op
                         (format "unknown argument kind: ~a" kind)
                         stx)]))

(define-syntax (define-generated-op stx)
  (syntax-parse stx
    [(_ name:id c-id:id ([arg:id kind:id] ...))
     (define-values (rev-specs rev-call-args)
       (for/fold ([specs '()] [call-args '()])
                 ([a (in-list (syntax->list #'(arg ...)))]
                  [k (in-list (syntax->datum #'(kind ...)))])
         (define-values (s c) (kind-pieces stx a k))
         (values (append (reverse s) specs)
                 (append (reverse c) call-args))))
     (with-syntax ([raw-name (format-id #'name "~a/raw" #'name)]
                   [(spec ...) (reverse rev-specs)]
                   [(call-arg ...) (reverse rev-call-args)])
       #'(begin
           (define-torch raw-name
             (_fun spec ... -> _Tensor/null)
             #:c-id c-id
             #:wrap (allocator tr-tensor-free/raw))
           (define (name arg ...)
             (wrap-tensor
              (check-handle 'name (raw-name call-arg ...))))))]))
