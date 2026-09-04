#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre)
         (only-in racket/contract/base
                  -> ->* any any/c cons/c contract-out listof or/c)
         (only-in racket/generic define-generics)
         (only-in racket/list append-map check-duplicates)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt" tensor?)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "parameter.rkt" Buffer? Parameter?))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:layer
         layer?
         layer-forward ;; noqa
         layer-parameters ;; noqa
         layer-named-parameters ;; noqa
         layer-buffers ;; noqa
         layer-named-children ;; noqa
         layer-set-training! ;; noqa
         layer-training? ;; noqa
         in-eval-mode
         define-layer)

(define-generics layer
  (layer-forward layer . inputs)
  (layer-parameters layer)
  (layer-named-parameters layer prefix)
  (layer-buffers layer)
  (layer-named-children layer)
  (layer-set-training! layer training?)
  (layer-training? layer))

(module+ checked
  (provide (contract-out [layer? (-> any/c boolean?)])))

;; Depth-first, own params before children's, in declaration order —
;; PyTorch's parameters() order, which seeded-init parity relies on.
(define/contract-out (parameters m) ;; noqa
  (-> layer? (listof tensor?))
  (layer-parameters m))

(define/checked-out (named-parameters m [prefix ""]) ;; noqa
  (->* [layer?] [string?] (listof (cons/c string? tensor?)))
  (layer-named-parameters m prefix))

(define/contract-out (buffers m) ;; noqa
  (-> layer? (listof tensor?))
  (layer-buffers m))

(define/contract-out (children m) ;; noqa
  (-> layer? (listof layer?))
  (map cdr (layer-named-children m)))

(define/contract-out (named-children m) ;; noqa
  (-> layer? (listof (cons/c string? layer?)))
  (layer-named-children m))

(define/contract-out (forward m . inputs) ;; noqa
  (-> layer? any/c ... any)
  (apply layer-forward m inputs))

(define/contract-out (train! m)
  (-> layer? layer?)
  (layer-set-training! m #t)
  m)

(define/contract-out (eval! m)
  (-> layer? layer?)
  (layer-set-training! m #f)
  m)

;; restores the aggregate prior mode tree-wide: a hand-mixed tree collapses
;; to all-train or all-eval on exit
(define/contract-out (call-with-eval-mode m thunk)
  (-> layer? (-> any) any)
  (define was-training? (layer-training? m))
  (dynamic-wind (lambda () (eval! m))
                thunk
                (lambda () (if was-training? (train! m) (eval! m)))))

(define-syntax-parse-rule (in-eval-mode m:expr body:expr ...+)
  (call-with-eval-mode m (lambda () body ...)))

(struct registry (forward params buffers children)
  #:property prop:procedure
  (lambda (self . inputs) (apply (registry-forward self) self inputs))
  #:methods gen:layer
  [(define (layer-forward self . inputs)
     (apply (registry-forward self) self inputs))
   (define (layer-parameters self)
     (append (map cdr (registry-params self))
             (append-map child-parameters (registry-children self))))
   (define (layer-named-parameters self prefix)
     (append (for/list ([p (in-list (registry-params self))])
               (cons (string-append prefix (car p)) (cdr p)))
             (append-map (lambda (c) (child-named-parameters c prefix))
                         (registry-children self))))
   (define (layer-buffers self)
     (append (map cdr (registry-buffers self))
             (append-map child-buffers (registry-children self))))
   (define (layer-named-children self)
     (registry-children self))
   (define (layer-set-training! self training?)
     (for ([c (in-list (registry-children self))])
       (child-set-training! c training?)))
   (define (layer-training? self)
     (andmap child-training? (registry-children self)))])

(define (child-parameters c)
  (layer-parameters (cdr c)))

(define (child-buffers c)
  (layer-buffers (cdr c)))

(define (child-set-training! c training?)
  (layer-set-training! (cdr c) training?))

(define (child-training? c)
  (layer-training? (cdr c)))

;; an empty name (a LayerList registered with #:prefix "") contributes no
;; segment: its children name themselves by index alone
(define (child-named-parameters c prefix)
  (layer-named-parameters (cdr c)
                          (if (string=? (car c) "")
                              prefix
                              (string-append prefix (car c) "."))))

(struct LayerList% registry (prefix)
  #:reflection-name 'LayerList)

(define (layer-list-forward self . _inputs)
  (raise-arguments-error 'LayerList
                         "not applicable; iterate with layer-list->list"
                         "layer list" self))

(define/checked-out (LayerList layers #:prefix [prefix #f])
  (->* [(listof layer?)] [#:prefix (or/c #f string?)] layer-list?)
  (LayerList% layer-list-forward '() '()
              (for/list ([m (in-list layers)] [i (in-naturals)])
                (cons (number->string i) m))
              prefix))

(define/contract-out layer-list? (-> any/c boolean?) LayerList%?)

(define/checked-out (layer-list->list ll) ;; noqa
  (-> layer-list? (listof layer?))
  (map cdr (registry-children ll)))

(define (classify who names vals) ;; noqa
  (define-values (params buffers children)
    (for/fold ([params '()] [buffers '()] [children '()]
               #:result (values (reverse params)
                                (reverse buffers)
                                (reverse children)))
              ([name (in-list names)] [v (in-list vals)])
      (cond
        [(Parameter? v) (values (cons (cons name v) params) buffers children)]
        [(Buffer? v) (values params (cons (cons name v) buffers) children)]
        [(layer? v)
         (define registered-as
           (if (LayerList%? v) (or (LayerList%-prefix v) name) name))
         (values params buffers (cons (cons registered-as v) children))]
        [else (values params buffers children)])))
  (define clash (check-duplicates (map car children)))
  (when clash
    (raise-arguments-error who "two children register under the same name"
                           "name" clash))
  (values params buffers children))

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

  (define (bound-in? id ids) ;; noqa
    (for/or ([other (in-list ids)])
      (bound-identifier=? id other))))

(define-syntax (define-layer stx)
  (syntax-parse stx
    [(_ name:id (field:ctor-formal ...)
        (~alt (~optional (~seq #:init init:init-formals init-body:expr ...))
              (~optional (~seq #:reflection-name reflect:expr))
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))) ...
        #:forward (input:id ...) body:expr ...+)
     (define field-ids (syntax->list #'(field.id ...)))
     (define init? (attribute init))
     (when init?
       (for ([f (in-list field-ids)]
             [bare? (in-list (attribute field.bare?))])
         (unless bare?
           (raise-syntax-error
            #f
            "with #:init, a field is a bare identifier; defaults and keywords belong to the #:init formals"
            stx f))))
     (define init-ids (if init? (syntax->list #'(init.id ...)) '()))
     (define struct-id (generate-temporary #'name))
     (define (accessor field-id)
       (format-id struct-id "~a-~a" struct-id field-id))
     (with-syntax ([sid struct-id]
                   [sid? (format-id struct-id "~a?" struct-id)]
                   [name? (format-id #'name "~a?" #'name)]
                   [reflect-name (or (attribute reflect) #'(quote name))]
                   [formals (if init?
                                #'init.formals
                                #'((~@ field.decl ...) ...))]
                   [(init-body ...) (if init? #'(init-body ...) #'())]
                   [(absent ...)
                    (filter (lambda (f) (not (bound-in? f init-ids)))
                            (if init? field-ids '()))]
                   [(field-name ...)
                    (for/list ([f (in-list field-ids)])
                      (symbol->string (syntax-e f)))]
                   [(field-acc ...) (map accessor field-ids)])
       (with-syntax ([export (contract-export stx #'name #'name?
                                              (attribute ctc)
                                              (attribute pred))])
         #'(begin
             (struct sid registry (field.id ...)
               #:reflection-name reflect-name)
             (define name? sid?)
             (define (forward-proc self . inputs)
               (let ([field.id (field-acc self)] ...)
                 (apply (lambda (input ...) body ...) inputs)))
             (define (name . formals)
               (let ([absent #f] ...)
                 init-body ...
                 (let-values ([(params buffers children)
                               (classify 'name
                                         '(field-name ...)
                                         (list field.id ...))])
                   (sid forward-proc params buffers children field.id ...))))
             export)))]))
