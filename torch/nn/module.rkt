#lang racket/base

;; The nn.Module system: `gen:module` is the interface, `define-module` the
;; Python-style sugar. Where Python registers fields at runtime via
;; __setattr__, the macro knows the field list at expansion time; models are
;; plain struct trees with no global store — drop one and the GC finalizers
;; release every native handle.

(require (for-syntax racket/base
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: syntax-parse patterns need many
                     ;; exported bindings only-in would strip
                     syntax/parse/pre)
         (only-in racket/generic define-generics define/generic)
         (only-in "../foreign.rkt" requires-grad!))

;; The noqa'd exports are define-generics/define-module expansions, which
;; raco review cannot see without macro expansion.
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
         module-set-training! ;; noqa
         module-training? ;; noqa
         train!
         eval!
         call-with-eval-mode
         in-eval-mode
         define-module)

(define-generics module
  (module-forward module . inputs)
  (module-parameters module)
  (module-named-parameters module prefix)
  (module-buffers module)
  ;; mode-sensitive leaves (dropout) hold the flag; structural modules recurse.
  (module-set-training! module training?)
  ;; a structural module is training iff all its submodules are, vacuously #t
  ;; with none (modules default to training).
  (module-training? module))

;; Depth-first, own params before submodules', in declaration order —
;; PyTorch's parameters() order, which seeded-init parity relies on.
(define (parameters m)
  (module-parameters m))

;; (name . tensor) pairs with dotted paths: '("fc1.weight" . #<tensor>).
(define (named-parameters m [prefix ""])
  (module-named-parameters m prefix))

(define (buffers m)
  (module-buffers m))

(define (forward m . inputs)
  (apply module-forward m inputs))

(define (train! m)
  (module-set-training! m #t)
  m)

(define (eval! m)
  (module-set-training! m #f)
  m)

;; Restores the *aggregate* prior mode tree-wide, so a hand-mixed tree (some
;; leaves manually flipped) collapses to all-train or all-eval on exit —
;; per-leaf snapshotting would be needed to preserve a mixed tree.
(define (call-with-eval-mode m thunk)
  (define was-training? (module-training? m))
  (dynamic-wind (lambda () (eval! m))
                thunk
                (lambda () (if was-training? (train! m) (eval! m)))))

(define-syntax-rule (in-eval-mode m body ...)
  (call-with-eval-mode m (lambda () body ...)))

(begin-for-syntax
  (define-syntax-class binding
    #:description "[id init-expr] binding"
    (pattern [id:id init:expr]))

  ;; `decl` is what the formal contributes to the public constructor's
  ;; argument list; `id` is the struct field it populates.
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

;; #:coerce rebinds ctor args (let* order) before any init sees them;
;; #:params are registered trainable tensors (requires-grad! is set after
;; init, so RNG-consuming inits match PyTorch); #:buffers are registered but
;; not trainable; #:submodules nest gen:module values.
(define-syntax (define-module stx)
  (syntax-parse stx
    [(_ name:id (formal:ctor-formal ...)
        (~alt (~optional (~seq #:coerce (coerce:binding ...)))
              (~optional (~seq #:params (param:binding ...)))
              (~optional (~seq #:buffers (buffer:binding ...)))
              (~optional (~seq #:submodules (sub:binding ...)))
              ;; resolved at compile time below (not via a template `~?`), so
              ;; its ellipsis depth here doesn't matter.
              (~optional (~seq #:reflection-name reflect:expr))) ...
        #:forward (input:id ...) body:expr ...+)
     (define (ids attr) (or attr '()))
     (define struct-id (generate-temporary #'name))
     (define reflect-stx (or (attribute reflect) #'(quote name)))
     (define (accessor field-id)
       (format-id struct-id "~a-~a" struct-id field-id))
     (define (name-string id)
       (symbol->string (syntax-e id)))
     (with-syntax ([sid struct-id]
                   [sid? (format-id struct-id "~a?" struct-id)]
                   [name? (format-id #'name "~a?" #'name)]
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
                ;; the generic forces a rest arg; the inner lambda restores
                ;; the declared #:forward arity (and its arity errors).
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
                ;; define-module structs are never mode-sensitive leaves.
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
                 (sid ctor-arg ... p ... b ... sm ...))))))]))
