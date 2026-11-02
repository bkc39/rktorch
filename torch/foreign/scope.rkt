#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre ...+ ~and ~bind ~or* expr))
         (only-in racket/contract/base -> any)
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "raw/scope.rkt" call-with-tensor-scope/handles scope-keep!)
         (only-in "structs.rkt" tensor-handle tensor-impl? tensor?))

(provide with-tensor-scope
         with-diag-scope
         diag-scope)

(define (parts v)
  (match v
    [(cons a d) (list a d)]
    [(? vector?) (vector->list v)]
    [(box b) (list b)]
    [(? hash?) (append (hash-keys v) (hash-values v))]
    [(? struct?) (cdr (vector->list (struct->vector v)))]
    [_ '()]))

(define (tensor-handles v)
  (define seen (make-hasheq))
  (let walk ([v v] [found '()])
    (cond
      [(tensor-impl? v) (cons (tensor-handle v) found)]
      [(hash-ref seen v #f) found]
      [else
       (hash-set! seen v #t)
       (for/fold ([found found]) ([p (in-list (parts v))])
         (walk p found))])))

(define/contract-out (call-with-tensor-scope thunk) ;; noqa
  (-> (-> any) any)
  (call-with-tensor-scope/handles thunk tensor-handles))

(define-syntax-parse-rule (with-tensor-scope body:expr ...+)
  (call-with-tensor-scope (lambda () body ...)))

(define/contract-out (tensor-keep t) ;; noqa
  (-> tensor? tensor?)
  (scope-keep! (tensor-handle t))
  t)

(define diag-scope
  (string->symbol (or (getenv "RKTORCH_DIAG_SCOPE") "off")))

(define-syntax-parse-rule (with-diag-scope (~or* (~and #:build (~bind [arms #'(build ref)]))
                                                 (~and #:ref (~bind [arms #'(ref)])))
                            body:expr ...+)
  (if (memq diag-scope 'arms)
      (with-tensor-scope body ...)
      (let () body ...)))
