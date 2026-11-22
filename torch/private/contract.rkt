#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in racket/contract/base contract-out unsupplied-arg?)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide define/contract-out
         define/checked-out
         supplied-or)

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
                          stx)))

  (define-splicing-syntax-class contracted-definition ;; noqa
    #:attributes (name contract definition)
    (pattern (~seq name:id contract:expr value:expr)
      #:with definition #'(define name value))
    (pattern (~seq (~and header (_ . _)) contract:expr body:expr ...+)
      #:with name (header-base #'header)
      #:with definition #'(define header body ...)))) ;; noqa

(define-syntax-parse-rule (define/contract-out d:contracted-definition) ;; noqa
  #:do [(module-level! this-syntax)]
  (begin
    d.definition
    (provide (contract-out [d.name d.contract]))
    (module+ unchecked (provide d.name))))

(define-syntax-parse-rule (define/checked-out d:contracted-definition) ;; noqa
  #:do [(module-level! this-syntax)]
  (begin
    d.definition
    (provide d.name)
    (module+ checked (provide (contract-out [d.name d.contract])))
    (module+ unchecked (provide d.name))))

(define (supplied-or v [default #f])
  (if (unsupplied-arg? v) default v))
