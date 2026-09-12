#lang racket/base

(require (for-template racket/base
                       (only-in racket/contract/base -> any/c contract-out))
         (only-in racket/syntax format-id)
         ;; whole-module on purpose: the syntax classes need bindings only-in
         ;; would strip
         syntax/parse/pre)

(provide contract-export
         ctor-formal
         init-formals)

(define (predicate-name name)
  (define downcased
    (for/fold ([acc '()] [prev #f] #:result (reverse acc))
              ([c (in-string (symbol->string (syntax-e name)))])
      (define word-break?
        (and prev
             (char-upper-case? c)
             (or (char-lower-case? prev) (char-numeric? prev))))
      (values (cons (char-downcase c) (if word-break? (cons #\- acc) acc))
              c)))
  (format-id name "~a?" (list->string downcased)))

(define (contract-export stx name name? contract predicate) ;; noqa
  (cond
    [contract
     (unless (eq? 'module (syntax-local-context))
       (raise-syntax-error
        #f
        "#:contract is only allowed at module level, since it expands to a `provide`"
        stx))
     (define pred-id (or predicate (predicate-name name)))
     (define alias? (not (eq? (syntax-e pred-id) (syntax-e name?))))
     (with-syntax ([name name] [name? name?] [name/lower pred-id]
                   [contract contract])
       (if alias?
           #'(begin
               (define name/lower (procedure-rename name? 'name/lower)) ;; noqa
               (provide (contract-out [name contract]
                                      [name/lower (-> any/c boolean?)])))
           #'(provide (contract-out [name contract]
                                    [name/lower (-> any/c boolean?)]))))]
    [predicate
     (raise-syntax-error
      #f "#:predicate names the exported predicate and needs #:contract"
      stx predicate)]
    [else #'(begin)]))

(define-splicing-syntax-class ctor-formal ;; noqa
  #:description
  "constructor formal (id, [id default], or #:kw id / #:kw [id default])"
  (pattern id:id
    #:attr bare? #t
    #:with (decl ...) #'(id))
  (pattern [id:id default:expr]
    #:attr bare? #f
    #:with (decl ...) #'([id default]))
  (pattern (~seq (~and kw:keyword (~not #:rest)) id:id)
    #:attr bare? #f
    #:with (decl ...) #'(kw id))
  (pattern (~seq (~and kw:keyword (~not #:rest)) [id:id default:expr])
    #:attr bare? #f
    #:with (decl ...) #'(kw [id default])))

(define-syntax-class init-formals
  #:description "#:init formals: (formal ... [#:rest id]) or (formal ... . id)"
  (pattern (f:ctor-formal ... #:rest rest:id)
    #:with (id ...) #'(f.id ... rest)
    #:with formals #'((~@ f.decl ...) ... . rest))
  (pattern (f:ctor-formal ... . rest:id)
    #:with (id ...) #'(f.id ... rest)
    #:with formals #'((~@ f.decl ...) ... . rest))
  (pattern (f:ctor-formal ...)
    #:with (id ...) #'(f.id ...)
    #:with formals #'((~@ f.decl ...) ...)))
