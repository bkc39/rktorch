#lang racket/base

(require (for-syntax racket/base
                     racket/syntax
                     syntax/parse/pre)
         (only-in ffi/unsafe _fun _int _int32 _int64 _double _list _stdbool)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/memory.rkt" tensor-allocator tensor-allocator/rng)
         (only-in "raw/syntax.rkt" _Tensor _Tensor/null define-torch)
         (only-in "structs.rkt" wrap-tensor))

(provide define-generated-op)

;; codes mirror the C tr_dtype enum; -1 is c10::nullopt
(define (opt-dtype->code d)
  (case d
    [(#f) -1]
    [(int64) 4]
    [(float32) 6]
    [(float64) 7]
    [else (error 'define-generated-op "unsupported optional dtype: ~e" d)]))

(define-for-syntax (kind-pieces stx arg kind)
  (define len-arg (format-id arg "~a-len" arg))
  (define has-arg (format-id arg "~a-has" arg))
  (case kind
    [(tensor) (values (list #`(#,arg : _Tensor)) (list arg))]
    [(optional-tensor)
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
             (list #`(or #,arg 0) #`(and #,arg #t)))]
    [(optional-int-array)
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
           (define (name arg ...)
             (check-ok (raw-name call-arg ...) 'name)
             recv)))]
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
