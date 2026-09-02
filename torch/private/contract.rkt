#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in racket/contract/base contract-out))

(provide define/contract-out
         define/checked-out)

(begin-for-syntax
  (define (header-base h)
    (syntax-parse h
      [n:id #'n]
      [(inner . _) (header-base #'inner)]))

  (define (module-level! stx) ;; noqa
    (unless (eq? 'module (syntax-local-context))
      (raise-syntax-error #f
                          (string-append
                           "only allowed at module level, since it expands to a"
                           " `provide`; an unexported definition needs no"
                           " contract boundary")
                          stx))))

(define-syntax (define/contract-out stx)
  (module-level! stx)
  (syntax-parse stx
    [(_ name:id contract:expr value:expr)
     #'(begin
         (define name value)
         (provide (contract-out [name contract])))]
    [(_ (~and header (_ . _)) contract:expr body ...+)
     #:with name (header-base #'header)
     #'(begin
         (define header body ...)
         (provide (contract-out [name contract])))]))

(define-syntax (define/checked-out stx)
  (module-level! stx)
  (syntax-parse stx
    [(_ name:id contract:expr value:expr)
     #'(begin
         (define name value)
         (provide name)
         (module+ checked (provide (contract-out [name contract]))))]
    [(_ (~and header (_ . _)) contract:expr body ...+)
     #:with name (header-base #'header)
     #'(begin
         (define header body ...)
         (provide name)
         (module+ checked (provide (contract-out [name contract]))))]))
