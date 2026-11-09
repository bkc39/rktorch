#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/list check-duplicates)
                     (only-in racket/match match)
                     (only-in racket/syntax format-id generate-temporary)
                     (only-in syntax/parse/experimental/eh ~eh-var)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre
                     (only-in "../private/definer.rkt"
                              contract-export entry-formals layer-field layer-fields
                              layer-options))
         (only-in racket/contract/base
                  -> ->* ->i and/c any any/c cons/c contract contract-out
                  contract? flat-named-contract listof not/c or/c
                  unsupplied-arg?)
         (only-in racket/list check-duplicates remove-duplicates)
         (only-in racket/stxparam syntax-parameterize)
         (only-in "../foreign.rkt" tensor?)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "buffer.rkt" Buffer?)
         (only-in "parameter.rkt" Parameter?)
         (only-in "private/field-walk.rkt" rebuilt-layer)
         (only-in "private/layer-containers.rkt"
                  Children% Children%-alist Children%? Fn% Parameters% Parameters%?
                  check-names fn-forward)
         (only-in "private/layer-core.rkt"
                  gen:layer layer-buffers layer-mode layer-named-buffers
                  layer-named-children layer-named-parameters layer-parameters
                  layer-rebuild-in layer-set-mode! layer? prop:on-move prop:rebuilder
                  registry registry? set-registry-mode! with-mode)
         (only-in "private/layer-fields.rkt"
                  assemble check-entry-arity entry-procedure layer-methods-of
                  layer-methods? no-forward prop:layer-methods write-field!))

(provide gen:layer
         layer?
         layer-parameters
         layer-named-parameters
         layer-buffers
         layer-named-buffers
         layer-named-children
         layer-mode
         layer-set-mode!
         registry?
         set-registry-mode!
         with-mode
         define-layer)

(define/checked-out mode/c contract?
  (flat-named-contract 'mode/c (or/c 'train 'eval)))

(define/checked-out (training? mode) ;; noqa
  (-> mode/c boolean?)
  (eq? mode 'train))

(define/contract-out (evaluating? mode) ;; noqa
  (-> mode/c boolean?)
  (eq? mode 'eval))

(define/checked-out (layer-rebuild m child tensor)
  (-> layer? (-> string? layer? layer?) (-> tensor? tensor?) layer?)
  (layer-rebuild-in m child tensor))

(module+ checked
  (provide (contract-out [layer? (-> any/c boolean?)])))

;; Depth-first, own params before children's, in declaration order —
;; PyTorch's parameters() order, which seeded-init parity relies on.
(define/contract-out (parameters m) ;; noqa
  (-> layer? (listof tensor?))
  (remove-duplicates (layer-parameters m) eq?))

(define/checked-out (named-parameters m [prefix ""]) ;; noqa
  (->* [layer?] [string?] (listof (cons/c string? tensor?)))
  (remove-duplicates (layer-named-parameters m prefix) eq? #:key cdr))

(define/contract-out (in-parameters m) ;; noqa
  (-> layer? sequence?)
  (make-do-sequence
   (lambda ()
     (values car cdr (parameters m) pair? #f #f))))

(define/contract-out (in-named-parameters m) ;; noqa
  (-> layer? sequence?)
  (make-do-sequence
   (lambda ()
     (values (lambda (entries) (values (caar entries) (cdar entries)))
             cdr
             (named-parameters m)
             pair?
             #f
             #f))))

(define/contract-out (buffers m) ;; noqa
  (-> layer? (listof tensor?))
  (remove-duplicates (layer-buffers m) eq?))

(define/checked-out (named-buffers m [prefix ""]) ;; noqa
  (->* [layer?] [string?] (listof (cons/c string? tensor?)))
  (remove-duplicates (layer-named-buffers m prefix) eq? #:key cdr))

(define/contract-out (children m) ;; noqa
  (-> layer? (listof layer?))
  (map cdr (named-children m)))

(define/contract-out (named-children m) ;; noqa
  (-> layer? (listof (cons/c string? layer?)))
  (remove-duplicates (layer-named-children m) eq? #:key cdr))

(define (as-layer v)
  (if (layer? v) v (procedure->Layer v)))

(define/checked-out step/c contract?
  (flat-named-contract 'step/c (or/c layer? procedure?)))

(define/checked-out child-name/c contract?
  (flat-named-contract 'child-name/c
                       (and/c string? (not/c "") (not/c #rx"[.]"))))

(define/contract-out (procedure->Layer proc
                                       #:parameters [params '()]
                                       #:buffers [bufs '()]
                                       #:children [kids '()])
  (->i ([proc procedure?])
       (#:parameters [params (listof (cons/c child-name/c Parameter?))]
        #:buffers [bufs (listof (cons/c child-name/c Buffer?))]
        #:children [kids (listof (cons/c child-name/c layer?))])
       #:pre (params bufs kids)
       (not (check-duplicates
             (map car (append (if (unsupplied-arg? params) '() params)
                              (if (unsupplied-arg? bufs) '() bufs)
                              (if (unsupplied-arg? kids) '() kids)))))
       [result layer?])
  (check-names 'procedure->Layer
               (Fn% fn-forward params bufs kids 'train proc)))

(define/contract-out Parameters? (-> any/c boolean?) Parameters%?)

(define/checked-out (parameters-by-key entries) ;; noqa
  (-> (listof (cons/c child-name/c Parameter?)) Parameters?)
  (Parameters% entries))

(define/contract-out Children? (-> any/c boolean?) Children%?)

(define/checked-out (children-by-index layers) ;; noqa
  (-> (listof step/c) Children?)
  (Children% (for/list ([m (in-list layers)] [i (in-naturals)])
               (cons (number->string i) (as-layer m)))))

(define/checked-out (children-by-key entries) ;; noqa
  (-> (listof (cons/c child-name/c step/c)) Children?)
  (Children% (for/list ([e (in-list entries)])
               (cons (car e) (as-layer (cdr e))))))

(define/checked-out (in-layers v) ;; noqa
  (-> (or/c Children? layer?) sequence?)
  (in-list (map cdr (if (layer? v) (layer-named-children v) (Children%-alist v)))))

(define/contract-out (child-ref m name) ;; noqa
  (-> layer? string? (or/c layer? #f))
  (define entry (assoc name (layer-named-children m)))
  (and entry (cdr entry)))

(module+ private
  (provide define-layer*)

  (define/contract-out (layer-method m name) ;; noqa
    (-> layer? symbol? (or/c procedure? #f))
    (and (layer-methods? m) (hash-ref (layer-methods-of m) name #f)))

  (define/contract-out (layer-methods m) ;; noqa
    (-> layer? (or/c hash? #f))
    (and (layer-methods? m) (layer-methods-of m))))

(begin-for-syntax
  (define ((with-mode-transformer self-id) stx) ;; noqa
    (syntax-parse stx
      [(_ id:id body:expr ...+)
       #`(let ([id (layer-mode #,self-id)]) body ...)]
      [(_ body:expr ...+)
       #`(let ([#,(datum->syntax stx 'mode) (layer-mode #,self-id)])
           body ...)]))

  (define-syntax-class (entry scope)
    #:attributes ((checker-def 1) who proc)
    (pattern (hidden:nat who:id (~var f (entry-formals "entry")) body:expr ...+)
      #:do [(define n (length (attribute f.id)))]
      #:fail-when (and (> (syntax-e #'hidden) n) #'hidden)
      "#:hidden-arguments counts more inputs than the positional formals"
      #:with ([checked ctc] ...)
      (for/list ([i (in-list (attribute f.id))] [c (in-list (attribute f.ctc))] #:when c)
        (list i c))
      #:with (checker ...) (generate-temporaries #'(checked ...))
      #:with (checker-def ...)
      #'((define checker
           (contract (-> ctc any) values 'who 'caller 'who (quote-syntax who)))
         ...)
      #:with (binder bindings) scope
      #:with lam
      (syntax-property
       #'(lambda (f.id ... f.kw-decl ... . (~? f.restarg ()))
           (let ([checked (checker checked)] ...) body ...))
       'inferred-name (syntax-e #'who))
      #:with proc
      #`(entry-procedure
         (lambda (self kws kw-args inputs)
           (check-entry-arity 'who #,n #,(and (attribute f.restarg) #t) hidden inputs)
           (binder bindings
             (syntax-parameterize ([with-mode (with-mode-transformer #'self)])
               (keyword-apply lam kws kw-args inputs)))))))

  (define ((field-transformer self acc mut who field) stx)
    (syntax-parse stx
      #:literals (set!)
      [(set! _ v:expr) #`(write-field! '#,who #,self '#,field #,acc #,mut v)]
      [(_ arg ...) #`((#,acc #,self) arg ...)]
      [_:id #`(#,acc #,self)]))

  (struct field-access (binder binding))

  (define immutable-fields
    (field-access #'let
                  (lambda (_who field acc _mut) #`[#,field (#,acc self)])))

  (define mutable-fields
    (field-access #'let-syntax
                  (lambda (who field acc mut)
                    #`[#,field (make-set!-transformer
                                (field-transformer (quote-syntax self)
                                                   (quote-syntax #,acc)
                                                   (quote-syntax #,mut)
                                                   '#,who '#,field))])))

  (define (field-ids struct-id fields pattern)
    (map (lambda (f) (format-id struct-id pattern struct-id f)) fields))

  (define (layer-expansion stx access)
    (syntax-parse stx
      [(_ name:id fields
          (~alt (~eh-var o layer-options)
                (~optional (~seq #:forward (~var fwd (entry-formals "forward")) fwd-body:expr ...+))
                (~seq #:method (~optional (~seq #:hidden-arguments hidden:nat))
                      (m-name:id . (~var m-formals (entry-formals "method"))) m-body:expr ...+)
                (~seq #:property prop:expr prop-val:expr)) ...)
       #:with (~var f (layer-fields (and (attribute o.init) "#:init"))) #'fields
       #:with sid (generate-temporary #'name)
       #:with sid? (format-id #'sid "~a?" #'sid)
       #:with (acc ...) (field-ids #'sid (attribute f.id) "~a-~a")
       #:with (mut ...) (field-ids #'sid (attribute f.id) "set-~a-~a!")
       #:with (field-name ...) (map (lambda (f) (symbol->string (syntax-e f))) (attribute f.id))
       #:with (field-spec ...) (map (lambda (f) #`[#,f #:mutable]) (attribute f.id))
       #:with (filled ...) (generate-temporaries (attribute f.id))
       #:with binder (field-access-binder access)
       #:with (binding ...)
       (for/list ([fid (in-list (attribute f.id))]
                  [a (in-list (attribute acc))]
                  [m (in-list (attribute mut))])
         ((field-access-binding access) #'name fid a m))
       #:with formals #'(~? o.init.formals ((~@ f.decl ...) ...))
       #:with (absent ...)
       (if (attribute o.init)
           (remove* (attribute o.init.id) (attribute f.id) bound-identifier=?)
           '())
       #:with ((~var fe (entry #'(binder (binding ...)))) ...)
       #'(~? ((0 name fwd fwd-body ...)) ())
       #:with ((~var me (entry #'(binder (binding ...)))) ...)
       #'(((~? hidden 0) m-name m-formals m-body ...) ...)
       #:fail-when (check-duplicates (attribute me.who) eq? #:key syntax-e)
       "a method is defined twice"
       #:with forward-expr (match (attribute fe.proc)
                             [(list p) p]
                             ['() #'(no-forward 'name)])
       #:with (methods-property ...)
       (if (null? (attribute me.who))
           #'()
           #'(#:property prop:layer-methods (hasheq (~@ 'me.who me.proc) ...)))
       #:with (moved-property ...)
       #'(~? (#:property prop:on-move
              (lambda (self) (binder (binding ...) (let () o.moved-body ...))))
             ())
       #:with name? (format-id #'name "~a?" #'name)
       #:with export (contract-export stx #'name #'name? (attribute o.ctc) (attribute o.pred))
       #'(begin
           (struct sid registry (field-spec ...)
             #:reflection-name (~? o.reflect 'name)
             #:property prop:rebuilder
             (lambda (self child tensor seen) (rebuild self child tensor seen))
             moved-property ...
             methods-property ...
             (~@ #:property prop prop-val) ...)
           (define name? sid?)
           fe.checker-def ... ...
           me.checker-def ... ...
           (define (rebuild self child tensor seen)
             (rebuilt-layer self
                            #:memo seen #:child child #:tensor tensor #:who 'name
                            #:make sid #:forward forward-proc
                            #:names '(field-name ...) #:values (list (acc self) ...)
                            #:fill (lambda (copy filled ...) (mut copy filled) ... (void))))
           (define forward-proc forward-expr)
           (define (name . formals)
             (let ([absent #f] ...)
               (~? (~@ o.init-body ...))
               (assemble 'name sid forward-proc 'train '(field-name ...) (list f.id ...))))
           export)])))

(define-syntax (define-layer stx)
  (syntax-parse stx
    [(_ _:id (_:layer-field ...) (~alt (~eh-var o layer-options)) ...
        #:forward (~var _ (entry-formals "forward")) body:expr ...+)
     (layer-expansion stx immutable-fields)]))

(define-syntax (define-layer* stx)
  (layer-expansion stx mutable-fields))
