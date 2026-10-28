#lang racket/base

(require (for-template racket/base
                       (only-in racket/contract/base -> any/c contract-out))
         (only-in racket/syntax format-id)
         (only-in syntax/parse/experimental/eh define-eh-alternative-set)
         ;; whole-module on purpose: the syntax classes need bindings only-in
         ;; would strip
         syntax/parse/pre)

(provide contract-export
         ctor-formal
         entry-formals
         field-of
         forward-formal
         forward-keyword
         init-formals
         layer-field
         layer-fields
         layer-options)

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

(define-syntax-class forward-formal ;; noqa
  #:description "forward formal (id or [id : contract])"
  (pattern id:id
    #:attr ctc #f)
  (pattern [id:id (~datum :) c:expr]
    #:attr ctc #'c))

(define-splicing-syntax-class forward-keyword ;; noqa
  #:description "keyword forward formal (#:kw id or #:kw [id default])"
  (pattern (~seq kw:keyword id:id)
    #:with (decl ...) #'(kw id))
  (pattern (~seq kw:keyword [id:id default:expr])
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

(define-splicing-syntax-class layer-field
  #:description "a layer field"
  #:attributes (id bare? (decl 1))
  (pattern f:ctor-formal
    #:fail-when
    (and (regexp-match? #rx"[.]" (symbol->string (syntax-e #'f.id))) #'f.id)
    "a field name may not contain a dot; it is one state-dict segment"
    #:with id #'f.id
    #:attr bare? (attribute f.bare?)
    #:with (decl ...) #'(f.decl ...)))

(define-syntax-class (layer-fields init-clause)
  #:description "a field list"
  #:attributes ((id 1) (decl 2))
  (pattern (f:layer-field ...)
    #:fail-when
    (and init-clause
         (for/or ([id (in-list (attribute f.id))] [bare? (in-list (attribute f.bare?))])
           (and (not bare?) id)))
    (format "with ~a, a field is a bare identifier; defaults and keywords belong to the ~a formals"
            init-clause init-clause)
    #:with (id ...) #'(f.id ...)
    #:with ((decl ...) ...) #'((f.decl ...) ...)))

(define-syntax-class (field-of fields type-name)
  #:description "a field name"
  (pattern h:id
    #:fail-unless (memq (syntax-e #'h) fields)
    (format "~a is not a field of ~a" (syntax-e #'h) type-name)))

(define-eh-alternative-set layer-options
  (pattern (~optional (~seq #:init init:init-formals init-body:expr ...)))
  (pattern (~optional (~seq #:reflection-name reflect:expr)))
  (pattern (~optional (~seq #:contract ctc:expr)))
  (pattern (~optional (~seq #:predicate pred:id)))
  (pattern (~optional (~seq #:on-move moved-body:expr ...+))))

(define-syntax-class (entry-formals what)
  #:description (format "~a formals" what)
  #:attributes ((id 1) (ctc 1) (kw-decl 1) restarg)
  (pattern ((~alt input:forward-formal kw-input:forward-keyword) ...
            . (~or* () restarg:id))
    #:with (id ...) #'(input.id ...)
    #:attr (ctc 1) (attribute input.ctc)
    #:with (kw-decl ...) #'((~@ kw-input.decl ...) ...)))
