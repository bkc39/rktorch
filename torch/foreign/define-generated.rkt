#lang racket/base

(require (for-syntax racket/base
                     racket/syntax
                     syntax/parse/pre)
         (only-in ffi/unsafe
                  _double _fun _int _int32 _int64 _list _ptr _stdbool)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in "error.rkt" check-handle check-ok check-outputs)
         (only-in "raw/memory.rkt"
                  tensor-allocator
                  tensor-allocator/outputs
                  tensor-allocator/outputs/no-retry
                  tensor-allocator/no-retry)
         (only-in "raw/random.rkt" _Generator/current)
         (only-in "raw/syntax.rkt" _Tensor _Tensor/null define-torch)
         (only-in "structs.rkt" wrap-tensor))

(provide define-generated-op)

;; codes mirror the C tr_dtype enum; -1 is c10::nullopt
(define (opt-dtype->code d)
  (case d
    [(#f) -1]
    [(int32) 3]
    [(int64) 4]
    [(float32) 6]
    [(float64) 7]
    [else (error 'define-generated-op "unsupported optional dtype: ~e" d)]))

(define-for-syntax (kind-pieces stx arg kind)
  (define len-arg (format-id arg "~a-len" arg))
  (define has-arg (format-id arg "~a-has" arg))
  (case kind
    [(tensor mutable-tensor) (values (list #`(#,arg : _Tensor)) (list arg))]
    [(optional-tensor)
     (values (list #`(#,arg : _Tensor/null)) (list arg))]
    [(scalar double) (values (list #`(#,arg : _double)) (list arg))]
    [(int64) (values (list #`(#,arg : _int64)) (list arg))]
    [(bool) (values (list #`(#,arg : _stdbool)) (list arg))]
    [(int-array)
     (values (list #`(#,arg : (_s64vector i)) #`(#,len-arg : _int64))
             (list #`(list->s64vector #,arg) #`(length #,arg)))]
    [(tensor-list mutable-tensor-list)
     (values (list #`(#,arg : (_list i _Tensor)) #`(#,len-arg : _int64))
             (list arg #`(length #,arg)))]
    [(scalar-list)
     (values (list #`(#,arg : (_list i _double)) #`(#,len-arg : _int64))
             (list arg #`(length #,arg)))]
    [(optional-tensor-list)
     (values (list #`(#,arg : (_list i _Tensor/null)) #`(#,len-arg : _int64))
             (list arg #`(length #,arg)))]
    [(optional-int64)
     (values (list #`(#,arg : _int64) #`(#,has-arg : _stdbool))
             (list #`(or #,arg 0) #`(and #,arg #t)))]
    [(optional-scalar optional-double)
     (values (list #`(#,arg : _double) #`(#,has-arg : _stdbool))
             (list #`(or #,arg 0.0) #`(and #,arg #t)))]
    [(optional-int-array)
     (values (list #`(#,arg : (_s64vector i)) #`(#,len-arg : _int64)
                   #`(#,has-arg : _stdbool))
             (list #`(list->s64vector (or #,arg '()))
                   #`(if (pair? #,arg) (length #,arg) 0)
                   #`(and (pair? #,arg) #t)))]
    [(optional-generator)
     (values (list #`(#,arg : _Generator/current)) (list arg))]
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

(begin-for-syntax
  (define-syntax-class signature
    #:attributes ([arg 1] [spec 1] [call-arg 1])
    (pattern ([arg:id kind:id] ...)
      #:do [(define-values (specs call-args)
              (build-pieces this-syntax
                            (syntax->list #'(arg ...))
                            (syntax->datum #'(kind ...))))]
      #:with (spec ...) specs
      #:with (call-arg ...) call-args))

  (define-syntax-class (argument-of args)
    #:description "one of the op's arguments"
    (pattern aligned:id
      #:fail-unless (member #'aligned args free-identifier=?)
      "the aligned list is one of the arguments"))

  (define-splicing-syntax-class retry
    #:attributes (single outputs)
    (pattern (~seq #:no-retry)
      #:with single #'tensor-allocator/no-retry
      #:with outputs #'tensor-allocator/outputs/no-retry)
    (pattern (~seq)
      #:with single #'tensor-allocator
      #:with outputs #'tensor-allocator/outputs))

  (define-splicing-syntax-class (results args)
    #:attributes ([out 1] [ret 1] wrap finish [extra 1])
    (pattern (~seq #:inplace)
      #:fail-when (null? args) "an in-place op takes its receiver first"
      #:with (out ...) #'()
      #:with (ret ...) #'(_int)
      #:attr wrap #f
      #:with finish #'status->receiver
      #:with (extra ...) (list (car args)))
    (pattern (~seq r:retry #:returns n:nat)
      #:fail-when (and (< (syntax-e #'n) 2) #'n)
      "a single return takes no #:returns clause"
      #:with (slot ...) (generate-temporaries (build-list (syntax-e #'n) values))
      #:with (out ...) #'((slot : (_ptr o _Tensor/null)) ...)
      #:with (ret ...) #'((rc : _int) -> (and (zero? rc) (list slot ...)))
      #:with wrap #'r.outputs
      #:with finish #'outputs->values
      #:with (extra ...) #'())
    (pattern (~seq r:retry #:returns-list (~var aligned (argument-of args)))
      #:with aligned-len (format-id #'aligned "~a-len" #'aligned)
      #:with (out ...) #'((slots : (_list o _Tensor/null aligned-len))
                          (slots-len : _int64 = aligned-len))
      #:with (ret ...) #'((rc : _int) -> (and (zero? rc) slots))
      #:with wrap #'r.outputs
      #:with finish #'outputs->list
      #:with (extra ...) #'())
    (pattern (~seq r:retry)
      #:with (out ...) #'()
      #:with (ret ...) #'(_Tensor/null)
      #:with wrap #'r.single
      #:with finish #'handle->tensor
      #:with (extra ...) #'())))

(define (status->receiver who rc receiver)
  (check-ok rc who)
  receiver)

(define (handle->tensor who handle)
  (wrap-tensor (check-handle who handle)))

(define (outputs->values who handles)
  (apply values (outputs->list who handles)))

(define (outputs->list who handles)
  (map wrap-tensor (check-outputs who handles)))

(define-syntax (define-generated-op stx)
  (syntax-parse stx
    [(_ name:id c-id:id shape ... sig:signature)
     #:with ((~var r (results (syntax->list #'(sig.arg ...))))) #'(shape ...)
     #:with raw-name (format-id #'name "~a/raw" #'name)
     #'(begin
         (define-torch raw-name
           (_fun sig.spec ... r.out ... -> r.ret ...)
           #:c-id c-id
           (~? (~@ #:wrap r.wrap)))
         (define (name sig.arg ...)
           (r.finish 'name (raw-name sig.call-arg ...) r.extra ...)))]))
