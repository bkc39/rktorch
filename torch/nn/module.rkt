#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre)
         (only-in racket/contract/base
                  -> ->* any any/c cons/c contract-out listof)
         (only-in racket/generic define-generics define/generic)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt" requires-grad! tensor?)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:module
         module?
         module-forward ;; noqa
         module-parameters ;; noqa
         module-named-parameters ;; noqa
         module-buffers ;; noqa
         module-set-training! ;; noqa
         module-training? ;; noqa
         in-eval-mode
         define-module)

(define-generics module
  (module-forward module . inputs)
  (module-parameters module)
  (module-named-parameters module prefix)
  (module-buffers module)
  (module-set-training! module training?)
  (module-training? module))

(module+ checked
  (provide (contract-out [module? (-> any/c boolean?)])))

;; Depth-first, own params before submodules', in declaration order —
;; PyTorch's parameters() order, which seeded-init parity relies on.
(define/contract-out (parameters m) ;; noqa
  (-> module? (listof tensor?))
  (module-parameters m))

(define/checked-out (named-parameters m [prefix ""]) ;; noqa
  (->* [module?] [string?] (listof (cons/c string? tensor?)))
  (module-named-parameters m prefix))

(define/contract-out (buffers m) ;; noqa
  (-> module? (listof tensor?))
  (module-buffers m))

(define/contract-out (forward m . inputs) ;; noqa
  (-> module? any/c ... any)
  (apply module-forward m inputs))

(define/contract-out (train! m)
  (-> module? module?)
  (module-set-training! m #t)
  m)

(define/contract-out (eval! m)
  (-> module? module?)
  (module-set-training! m #f)
  m)

;; restores the aggregate prior mode tree-wide: a hand-mixed tree collapses
;; to all-train or all-eval on exit
(define/contract-out (call-with-eval-mode m thunk)
  (-> module? (-> any) any)
  (define was-training? (module-training? m))
  (dynamic-wind (lambda () (eval! m))
                thunk
                (lambda () (if was-training? (train! m) (eval! m)))))

(define-syntax-parse-rule (in-eval-mode m:expr body:expr ...+)
  (call-with-eval-mode m (lambda () body ...)))

(begin-for-syntax
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

  (define-syntax-class binding
    #:description "[id init-expr] binding"
    (pattern [id:id init:expr]))

  (define-splicing-syntax-class ctor-formal
    #:description
    "constructor formal (id, [id default], or #:kw id / #:kw [id default])"
    (pattern id:id
      #:with (decl ...) #'(id))
    (pattern [id:id default:expr]
      #:with (decl ...) #'([id default]))
    (pattern (~seq kw:keyword id:id)
      #:with (decl ...) #'(kw id))
    (pattern (~seq kw:keyword [id:id default:expr])
      #:with (decl ...) #'(kw [id default]))))

(define-syntax (define-module stx)
  (syntax-parse stx
    [(_ name:id (formal:ctor-formal ...)
        (~alt (~optional (~seq #:coerce (coerce:binding ...)))
              (~optional (~seq #:params (param:binding ...)))
              (~optional (~seq #:buffers (buffer:binding ...)))
              (~optional (~seq #:submodules (sub:binding ...)))
              (~optional (~seq #:reflection-name reflect:expr))
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))) ...
        #:forward (input:id ...) body:expr ...+)
     (define (ids attr) (or attr '()))
     (define struct-id (generate-temporary #'name))
     (define reflect-stx (or (attribute reflect) #'(quote name)))
     (define predicate-id (format-id #'name "~a?" #'name))
     (define (accessor field-id)
       (format-id struct-id "~a-~a" struct-id field-id))
     (define (name-string id)
       (symbol->string (syntax-e id)))
     (with-syntax ([sid struct-id]
                   [sid? (format-id struct-id "~a?" struct-id)]
                   [name? predicate-id]
                   [export (contract-export stx #'name predicate-id
                                            (attribute ctc) (attribute pred))]
                   [reflect-name reflect-stx]
                   [(ctor-arg ...) #'(formal.id ...)]
                   [(c ...) (ids (attribute coerce.id))]
                   [(c-init ...) (ids (attribute coerce.init))]
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
               #:reflection-name reflect-name
               #:property prop:procedure
               (lambda (self . inputs) (apply module-forward self inputs))
               #:methods gen:module
               [(define/generic recur-parameters module-parameters)
                (define/generic recur-named module-named-parameters)
                (define/generic recur-buffers module-buffers)
                (define/generic recur-set-training! module-set-training!)
                (define/generic recur-training? module-training?)
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
                          (recur-buffers (sm-acc self)) ...))
                (define (module-set-training! self training?)
                  (recur-set-training! (sm-acc self) training?) ...
                  (void))
                (define (module-training? self)
                  (and (recur-training? (sm-acc self)) ... #t))])
             (define name? sid?)
             (define (name (~@ formal.decl ...) ...)
               (let* ([c c-init] ...
                      [p (requires-grad! p-init)] ...
                      [b b-init] ...
                      [sm sm-init] ...)
                 (sid ctor-arg ... p ... b ... sm ...)))
             export)))]))
