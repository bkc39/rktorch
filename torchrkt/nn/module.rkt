#lang racket/base

;; The nn.Module system: a module is a named container of parameters +
;; buffers + submodules plus a forward function (the PyTorch framing).
;;
;; `gen:module` is the interface; `define-module` is the Python-style sugar.
;; Where Python registers fields at runtime by overriding __setattr__, the
;; macro knows the field list at expansion time, so registration is purely
;; structural: each module is a plain Racket struct holding its tensors and
;; submodules, and `parameters` recursively flattens the tree. There is no
;; global store — drop the model and the GC finalizers release every native
;; handle.

(require (for-syntax racket/base
                     racket/syntax
                     syntax/parse/pre)
         racket/generic
         "../foreign.rkt")

;; The module-* generics are defined by define-generics, which raco review
;; cannot see without expansion.
(provide gen:module
         module?
         module-forward ;; noqa
         module-parameters ;; noqa
         module-named-parameters ;; noqa
         module-buffers ;; noqa
         parameters
         named-parameters
         buffers
         forward
         define-module)

(define-generics module
  (module-forward module . inputs)
  (module-parameters module)
  (module-named-parameters module prefix)
  (module-buffers module))

;; Trainable tensors, depth-first: own params first, then each submodule's,
;; in declaration order — PyTorch's parameters() order, which seeded-init
;; parity (and therefore the optimizer walk) relies on.
(define (parameters m)
  (module-parameters m))

;; (name . tensor) pairs with dotted paths, like named_parameters():
;; '("fc1.weight" . #<tensor>).
(define (named-parameters m [prefix ""])
  (module-named-parameters m prefix))

(define (buffers m)
  (module-buffers m))

(define (forward m . inputs)
  (apply module-forward m inputs))

(begin-for-syntax
  (define-syntax-class binding
    #:description "[id init-expr] binding"
    (pattern [id:id init:expr])))

;; (define-module name (ctor-arg ...)
;;   #:params     ([p init] ...)     ; optional — registered trainable tensors
;;   #:buffers    ([b init] ...)     ; optional — registered, not trainable
;;   #:submodules ([m init] ...)     ; optional — nested gen:module values
;;   #:forward (input ...) body ...)
;;
;; Expands to a struct (one field per ctor-arg/param/buffer/submodule), a
;; gen:module implementation whose forward body sees every field by name, a
;; prop:procedure so (net x) applies forward, a `name?` predicate, and a
;; constructor `name` that evaluates the inits left-to-right (params get
;; requires-grad! set after init, so RNG-consuming inits match PyTorch).
(define-syntax (define-module stx)
  (syntax-parse stx
    [(_ name:id (ctor-arg:id ...)
        (~alt (~optional (~seq #:params (param:binding ...)))
              (~optional (~seq #:buffers (buffer:binding ...)))
              (~optional (~seq #:submodules (sub:binding ...)))) ...
        #:forward (input:id ...) body:expr ...+)
     (define (ids attr) (or attr '()))
     (define struct-id (generate-temporary #'name))
     (define (accessor field-id)
       (format-id struct-id "~a-~a" struct-id field-id))
     (define (name-string id)
       (symbol->string (syntax-e id)))
     (with-syntax ([sid struct-id]
                   [sid? (format-id struct-id "~a?" struct-id)]
                   [name? (format-id #'name "~a?" #'name)]
                   [(p ...) (ids (attribute param.id))]
                   [(p-init ...) (ids (attribute param.init))]
                   [(b ...) (ids (attribute buffer.id))]
                   [(b-init ...) (ids (attribute buffer.init))]
                   [(sm ...) (ids (attribute sub.id))]
                   [(sm-init ...) (ids (attribute sub.init))])
       (with-syntax ([(arg-acc ...) (map accessor (syntax->list #'(ctor-arg ...)))]
                     [(p-acc ...) (map accessor (syntax->list #'(p ...)))]
                     [(b-acc ...) (map accessor (syntax->list #'(b ...)))]
                     [(sm-acc ...) (map accessor (syntax->list #'(sm ...)))]
                     [(p-name ...) (map name-string (syntax->list #'(p ...)))]
                     [(sm-name ...) (map name-string (syntax->list #'(sm ...)))])
         #'(begin
             (struct sid (ctor-arg ... p ... b ... sm ...)
               #:reflection-name 'name
               #:property prop:procedure
               (lambda (self . inputs) (apply module-forward self inputs))
               #:methods gen:module
               [(define/generic recur-parameters module-parameters)
                (define/generic recur-named module-named-parameters)
                (define/generic recur-buffers module-buffers)
                ;; The generic is declared (module-forward module . inputs),
                ;; so the method must accept a rest too; the inner lambda
                ;; restores the declared #:forward arity (and its arity
                ;; errors).
                (define (module-forward self . inputs)
                  (let ([ctor-arg (arg-acc self)] ...
                        [p (p-acc self)] ...
                        [b (b-acc self)] ...
                        [sm (sm-acc self)] ...)
                    (apply (lambda (input ...) body ...) inputs)))
                (define (module-parameters self)
                  (append (list (p-acc self) ...)
                          (recur-parameters (sm-acc self)) ...))
                (define (module-named-parameters self prefix)
                  (append (list (cons (string-append prefix p-name)
                                      (p-acc self))
                                ...)
                          (recur-named (sm-acc self)
                                       (string-append prefix sm-name "."))
                          ...))
                (define (module-buffers self)
                  (append (list (b-acc self) ...)
                          (recur-buffers (sm-acc self)) ...))])
             (define name? sid?)
             (define (name ctor-arg ...)
               (let* ([p (requires-grad! p-init)] ...
                      [b b-init] ...
                      [sm sm-init] ...)
                 (sid ctor-arg ... p ... b ... sm ...))))))]))
