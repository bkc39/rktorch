#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/list check-duplicates)
                     (only-in racket/match match)
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre
                     (only-in "../private/definer.rkt"
                              contract-export ctor-formal forward-formal
                              forward-keyword init-formals))
         (only-in racket/contract/base
                  -> ->* ->i and/c any any/c cons/c contract contract-out
                  contract? flat-named-contract listof not/c or/c
                  unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in racket/list append-map check-duplicates remove-duplicates)
         (only-in racket/stxparam define-syntax-parameter syntax-parameterize)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt"
                  device dtype prop:to tensor?)
         (only-in (submod "../foreign.rkt" unsafe) to!)
         (only-in "../foreign/autograd-ops.rkt" collect-at-forward-trough!)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "buffer.rkt" Buffer?)
         (only-in "parameter.rkt" Parameter?))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:layer
         layer?
         layer-parameters ;; noqa
         layer-named-parameters ;; noqa
         layer-buffers ;; noqa
         layer-named-buffers ;; noqa
         layer-named-children ;; noqa
         layer-mode ;; noqa
         layer-set-mode! ;; noqa
         layer-rebuild ;; noqa
         registry?
         set-registry-mode!
         with-mode
         define-layer)

(define-generics layer
  (layer-forward layer . inputs)
  (layer-parameters layer)
  (layer-named-parameters layer prefix)
  (layer-buffers layer)
  (layer-named-buffers layer prefix)
  (layer-named-children layer)
  (layer-own-tensors layer)
  (layer-mode layer)
  (layer-set-mode! layer mode)
  (layer-rebuild layer child tensor)
  #:derive-property prop:to (lambda (m dev dtype) (move-layer! m dev dtype))
  #:derive-property prop:procedure
  (make-keyword-procedure
   (lambda (kws kw-args m . inputs) (call-layer/keywords m kws kw-args inputs))
   (lambda (m . inputs) (apply call-layer m inputs)))
  #:fallbacks
  [(define (layer-parameters self) '()) ;; noqa
   (define (layer-named-parameters self prefix) '()) ;; noqa
   (define (layer-buffers self) '()) ;; noqa
   (define (layer-named-buffers self prefix) '()) ;; noqa
   (define (layer-named-children self) '()) ;; noqa
   (define (layer-own-tensors self) (own-tensors self)) ;; noqa
   (define (layer-mode self) 'train) ;; noqa
   (define (layer-set-mode! self mode) (void)) ;; noqa
   (define (layer-rebuild self child tensor) ;; noqa
     (raise-arguments-error
      'layer-copy
      "a hand-written gen:layer has no layer-rebuild method to copy it with"
      "layer" self))])

(define/checked-out mode/c contract?
  (flat-named-contract 'mode/c (or/c 'train 'eval)))

(define/checked-out (training? mode) ;; noqa
  (-> mode/c boolean?)
  (eq? mode 'train))

(define/contract-out (evaluating? mode) ;; noqa
  (-> mode/c boolean?)
  (eq? mode 'eval))

(module+ checked
  (provide (contract-out [layer? (-> any/c boolean?)])))

;; PyTorch: "nn.Module.to only accepts floating point or complex dtypes",
;; and its convert forwards the dtype only to floating-point tensors — an
;; int64 or bool buffer keeps its dtype and changes device alone
(define floating-dtypes '(float32 float64 float16 bfloat16))

(define (floating? t)
  (and (memq (dtype t) floating-dtypes) #t))

(define-values (prop:on-move on-move? on-move-ref)
  (make-struct-type-property 'on-move))

(define (placement t)
  (cons (device t) (dtype t)))

(define (move-tensor! t dev dtype)
  (define dt (and dtype (floating? t) dtype))
  (cond
    [(and dev dt) (to! t dev dt)]
    [dev (to! t dev)]
    [dt (to! t dt)]
    [else (void)]))

;; The fallback for a layer that does not declare its own tensors apart from
;; its children's: both lists here are recursive, so this is the expensive
;; way round and `registry` overrides it.
(define (own-tensors m)
  (define theirs (make-hasheq))
  (for* ([c (in-list (layer-named-children m))]
         [t (in-list (append (layer-parameters (cdr c))
                             (layer-buffers (cdr c))))])
    (hash-set! theirs t #t))
  (for/list ([t (in-list (append (parameters m) (buffers m)))]
             #:unless (hash-ref theirs t #f))
    t))

(define (tree-placements m)
  (define seen (make-hasheq))
  (define before (make-hasheq))
  (let walk ([m m])
    (cond
      [(hash-has-key? seen m) (void)]
      [else
       (hash-set! seen m #t)
       (for ([c (in-list (layer-named-children m))])
         (walk (cdr c)))
       (for ([t (in-list (layer-own-tensors m))])
         (hash-set! before t (placement t)))]))
  before)

;; a tensor tied across siblings moves with the first, so read placements first
(define (move-layer! m dev dtype)
  (when (and dtype (not (memq dtype floating-dtypes)))
    (raise-arguments-error 'to "a layer only moves to a floating-point dtype"
                           "dtype" dtype))
  (move-tree! m dev dtype (and (hooked? m) (tree-placements m)) (make-hasheq))
  m)

(define (hooked? m)
  (define seen (make-hasheq))
  (let walk ([m m])
    (cond
      [(hash-has-key? seen m) #f]
      [else
       (hash-set! seen m #t)
       (or (on-move? m)
           (for/or ([c (in-list (layer-named-children m))])
             (walk (cdr c))))])))

(define (move-tree! m dev dtype before seen)
  (cond
    [(hash-has-key? seen m) (hash-ref seen m)]
    [else
     (define children-moved?
       (for/fold ([moved? #f]) ([c (in-list (layer-named-children m))])
         (or (move-tree! (cdr c) dev dtype before seen) moved?)))
     (define own (layer-own-tensors m))
     (for ([t (in-list own)])
       (move-tensor! t dev dtype))
     (define moved?
       (and before
            (or children-moved?
                (for/or ([t (in-list own)])
                  (not (equal? (hash-ref before t) (placement t)))))))
     (hash-set! seen m moved?)
     (when (and moved? (on-move? m))
       ((on-move-ref m) m))
     moved?]))

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

(define-syntax-parameter with-mode
  (lambda (stx)
    (raise-syntax-error
     #f "only allowed inside a define-layer #:forward body" stx)))

(begin-for-syntax
  (define ((with-mode-transformer self-id) stx) ;; noqa
    (syntax-parse stx
      [(_ id:id body:expr ...+)
       #`(let ([id (layer-mode #,self-id)]) body ...)]
      [(_ body:expr ...+)
       #`(let ([#,(datum->syntax stx 'mode) (layer-mode #,self-id)])
           body ...)])))

(define layer-call-key (make-continuation-mark-key 'layer-call))

;; Applying any layer reaches the method through here, since gen:layer
;; derives prop:procedure from it, so a layer written by hand gets the
;; trough too. The mark tells a nested call from the outermost one, whose
;; return is where a no-grad loop's memory is at its lowest.
(define-syntax-parse-rule (at-forward-trough call:expr)
  (if (continuation-mark-set-first #f layer-call-key)
      call
      (begin0 (with-continuation-mark layer-call-key #t call)
              (collect-at-forward-trough!))))

(define (call-layer m . inputs) ;; noqa
  (at-forward-trough (apply layer-forward m inputs)))

;; Only define-layer's forward and procedure->Layer's take keywords; a
;; hand-written layer-forward is a generic method, which cannot, so the
;; application reports that method as the one refusing them.
(define (call-layer/keywords m kws kw-args inputs) ;; noqa
  (define forward (if (registry? m) (registry-forward m) layer-forward))
  (at-forward-trough (keyword-apply forward kws kw-args m inputs)))

(struct registry (forward params buffers children [mode #:mutable])
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
   (define (layer-named-buffers self prefix)
     (append (for/list ([b (in-list (registry-buffers self))])
               (cons (string-append prefix (car b)) (cdr b)))
             (append-map (lambda (c) (child-named-buffers c prefix))
                         (registry-children self))))
   (define (layer-named-children self)
     (registry-children self))
   (define (layer-own-tensors self)
     (append (map cdr (registry-params self))
             (map cdr (registry-buffers self))))
   (define (layer-set-mode! self mode)
     (set-registry-mode! self mode)
     (for ([c (in-list (registry-children self))])
       (child-set-mode! c mode)))
   (define (layer-mode self)
     (registry-mode self))
   (define (layer-rebuild self child tensor)
     ((rebuilder-of self) self child tensor))])

(define-values (prop:rebuilder rebuilder? rebuilder-of) ;; noqa
  (make-struct-type-property 'rebuilder))

(define (rebuilt-value name v child tensor) ;; noqa
  (let walk ([v v])
    (cond
      [(layer? v) (child name v)]
      [(tensor? v) (tensor v)]
      [(Parameters%? v)
       (Parameters% (for/list ([e (in-list (Parameters%-alist v))])
                      (cons (car e) (tensor (cdr e)))))]
      [(Children%? v)
       (Children% (for/list ([e (in-list (Children%-alist v))])
                    (cons (car e) (child (car e) (cdr e)))))]
      [(pair? v) (cons (walk (car v)) (walk (cdr v)))]
      [(vector? v)
       (define copy (for/vector #:length (vector-length v)
                                ([x (in-vector v)])
                      (walk x)))
       (if (immutable? v) (vector->immutable-vector copy) copy)]
      [(box? v) (box (walk (unbox v)))]
      [else v])))

(define (child-parameters c)
  (layer-parameters (cdr c)))

(define (child-buffers c)
  (layer-buffers (cdr c)))

(define (child-set-mode! c mode)
  (layer-set-mode! (cdr c) mode))

(define (child-prefix c prefix)
  (if (string=? (car c) "") prefix (string-append prefix (car c) ".")))

(define (child-named-parameters c prefix)
  (layer-named-parameters (cdr c) (child-prefix c prefix)))

(define (child-named-buffers c prefix)
  (layer-named-buffers (cdr c) (child-prefix c prefix)))

(struct Fn% registry (proc)
  #:reflection-name 'Fn
  #:property prop:rebuilder
  (lambda (self _child _tensor)
    (unless (null? (append (registry-params self) (registry-buffers self)
                           (registry-children self)))
      (raise-arguments-error
       'layer-copy
       (string-append "a procedure->Layer layer with parameters, buffers or"
                      " children cannot be copied: its procedure would"
                      " still read the original's")
       "layer" self))
    (Fn% fn-forward '() '() '() (registry-mode self) (Fn%-proc self))))

(define fn-forward
  (make-keyword-procedure
   (lambda (kws kw-args self . inputs)
     (keyword-apply (Fn%-proc self) kws kw-args inputs))
   (lambda (self . inputs)
     (apply (Fn%-proc self) inputs))))

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

(struct Parameters% (alist)
  #:reflection-name 'Parameters)

(define/contract-out Parameters? (-> any/c boolean?) Parameters%?)

(define/checked-out (parameters-by-key entries) ;; noqa
  (-> (listof (cons/c child-name/c Parameter?)) Parameters?)
  (Parameters% entries))

(struct Children% (alist)
  #:reflection-name 'Children)

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

(define (classify names vals) ;; noqa
  (for/fold ([params '()] [buffers '()] [children '()]
             #:result (values (reverse params)
                              (reverse buffers)
                              (reverse children)))
            ([name (in-list names)] [v (in-list vals)])
    (cond
      [(Parameter? v) (values (cons (cons name v) params) buffers children)]
      [(Parameters? v)
       (values (append (reverse (Parameters%-alist v)) params)
               buffers children)]
      [(Buffer? v) (values params (cons (cons name v) buffers) children)]
      [(Children? v)
       (values params buffers
               (append (reverse (Children%-alist v)) children))]
      [(layer? v) (values params buffers (cons (cons name v) children))]
      [else (values params buffers children)])))

(define (check-names who m) ;; noqa
  (define child-clash (check-duplicates (map car (layer-named-children m))))
  (when child-clash
    (raise-arguments-error who "two children would share a name"
                           "name" child-clash))
  (define params (map car (layer-named-parameters m "")))
  (define clash (check-duplicates params))
  (when clash
    (raise-arguments-error who "two parameters would share a name"
                           "name" clash))
  (define entry-clash
    (check-duplicates (append params (map car (layer-named-buffers m "")))))
  (when entry-clash
    (raise-arguments-error who "two state-dict entries would share a name"
                           "name" entry-clash))
  m)

(struct registry* registry (fixed))

(define-values (prop:layer-methods layer-methods? layer-methods-of)
  (make-struct-type-property 'layer-methods))

(define (layer-methods m)
  (and (layer-methods? m) (layer-methods-of m)))

(define (layer-method m name)
  (and (layer-methods? m) (hash-ref (layer-methods-of m) name #f)))

(module+ private
  (provide define-layer*
           (contract-out
            [layer-method (-> layer? symbol? (or/c procedure? #f))]
            [layer-methods (-> layer? (or/c hash? #f))])))

(define (registered? v)
  (or (Parameter? v) (Parameters? v) (Buffer? v) (Children? v) (layer? v)))

(define (registered-names names vals) ;; noqa
  (for/list ([name (in-list names)] [v (in-list vals)] #:when (registered? v))
    name))

(define (classify-kept kept names vals) ;; noqa
  (define entries
    (for/list ([name (in-list names)] [v (in-list vals)] #:when (member name kept))
      (cons name v)))
  (classify (map car entries) (map cdr entries)))

(define (write-field! who self field set v) ;; noqa
  (when (member field (registry*-fixed self))
    (raise-arguments-error
     who
     "set! cannot replace a field that held a layer, parameter or buffer at construction"
     "field" (string->symbol field)))
  (set self v))

(define (check-entry-arity who enough? n expected inputs) ;; noqa
  (unless (enough? (length inputs) n)
    (apply raise-arity-error who expected inputs)))

(define (entry-procedure run) ;; noqa
  (make-keyword-procedure
   (lambda (kws kw-args self . inputs) (run self kws kw-args inputs))
   (lambda (self . inputs) (run self '() '() inputs))))

(define (no-forward who) ;; noqa
  (entry-procedure
   (lambda (self _kws _kw-args _inputs)
     (raise-arguments-error who "this layer has no forward, so it cannot be applied"
                            "layer" self))))

(begin-for-syntax
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

  (define-syntax-class entry-formals
    #:attributes ((id 1) (ctc 1) (kw-decl 1) restarg)
    (pattern ((~alt input:forward-formal kw-input:forward-keyword) ...
              . (~or* () restarg:id))
      #:with (id ...) #'(input.id ...)
      #:attr (ctc 1) (attribute input.ctc)
      #:with (kw-decl ...) #'((~@ kw-input.decl ...) ...)))

  (define-syntax-class (entry bind)
    #:attributes ((checker-def 1) who proc)
    (pattern (who:id f:entry-formals body:expr ...+)
      #:do [(define ctcs (attribute f.ctc))
            (define checkers
              (for/list ([c (in-list ctcs)]) (and c (generate-temporary #'check))))
            (define n (length (attribute f.id)))
            (define rest? (and (attribute f.restarg) #t))]
      #:with (checker-def ...)
      (for/list ([c (in-list ctcs)] [cid (in-list checkers)] #:when c)
        #`(define #,cid
            (contract (-> #,c any) values 'who 'caller 'who (quote-syntax who))))
      #:with (checked-binding ...)
      (for/list ([c (in-list ctcs)]
                 [cid (in-list checkers)]
                 [i (in-list (attribute f.id))]
                 #:when c)
        #`[#,i (#,cid #,i)])
      #:with lam
      (syntax-property
       (if rest?
           #'(lambda (f.id ... f.kw-decl ... . f.restarg)
               (let (checked-binding ...) body ...))
           #'(lambda (f.id ... f.kw-decl ...)
               (let (checked-binding ...) body ...)))
       'inferred-name (syntax-e #'who))
      #:with proc
      #`(entry-procedure
         (lambda (self kws kw-args inputs)
           (check-entry-arity 'who #,(if rest? #'>= #'=) #,n
                              #,(if rest? #`(arity-at-least #,n) n) inputs)
           #,(bind #'(syntax-parameterize ([with-mode (with-mode-transformer #'self)])
                       (keyword-apply lam kws kw-args inputs)))))))

  (define ((field-transformer read write) stx) ;; noqa
    (syntax-parse stx
      #:literals (set!)
      [(set! _ v:expr) #`(#,@(syntax->list write) v)]
      [(_ arg ...) #`(#,read arg ...)]
      [_:id read]))

  (define (unbare-field init? fields bares)
    (and init?
         (for/or ([f (in-list fields)] [bare? (in-list bares)])
           (and (not bare?) f))))

  (define unbare-message
    (string-append "with #:init, a field is a bare identifier;"
                   " defaults and keywords belong to the #:init formals"))

  (define (init-parts fields decls init-formals init-ids)
    (if init-formals
        (list init-formals
              (filter (lambda (f) (not (member f init-ids bound-identifier=?))) fields))
        (list (apply append decls) '())))

  (define (field-ids struct-id fields pattern)
    (for/list ([f (in-list fields)]) (format-id struct-id pattern struct-id f)))

  (define (layer-expansion stx star? ctc pred parts)
    (syntax-parse parts
      [(name formals (field ...) (assign ...) (absent ...) reflect-name
        (moved-body ...) (fwd ...) (method ...) ((prop prop-val) ...))
       #:do [(define fields (syntax->list #'(field ...)))
             (define struct-id (generate-temporary #'name))
             (define accs (field-ids struct-id fields "~a-~a"))
             (define names
               (for/list ([f (in-list fields)]) (symbol->string (syntax-e f))))
             (define bindings
               (for/list ([f (in-list fields)]
                          [acc (in-list accs)]
                          [mut (in-list (field-ids struct-id fields "set-~a-~a!"))]
                          [s (in-list names)])
                 (if star?
                     #`[#,f (make-set!-transformer
                             (field-transformer
                              (quote-syntax (#,acc self))
                              (quote-syntax (write-field! 'name self #,s #,mut))))]
                     #`[#,f (#,acc self)])))
             (define (bind e)
               #`(#,(if star? #'let-syntax #'let) #,bindings #,e))]
       #:with ((~var fe (entry bind)) ...) #'((name . fwd) ...)
       #:with ((~var me (entry bind)) ...) #'(method ...)
       #:fail-when (check-duplicates (attribute me.who) eq? #:key syntax-e)
       "a method is defined twice"
       #:with sid struct-id
       #:with sid? (format-id struct-id "~a?" struct-id)
       #:with name? (format-id #'name "~a?" #'name)
       #:with (field-name ...) names
       #:with (field-acc ...) accs
       #:with forward-expr (match (attribute fe.proc)
                             [(list p) p]
                             ['() #'(no-forward 'name)])
       #:with (field-spec ...) (if star? #'([field #:mutable] ...) #'(field ...))
       #:with parent (if star? #'registry* #'registry)
       #:with (method-def ...)
       (if star? #'((define methods (hasheq (~@ 'me.who me.proc) ...))) #'())
       #:with (prop-clause ...)
       (if star?
           #'(#:property prop:layer-methods methods
              (~@ #:property prop prop-val) ...)
           #'())
       #:with (moved-clause ...)
       (if (null? (attribute moved-body))
           #'()
           #`(#:property prop:on-move
              (lambda (self) #,(bind #'(let () moved-body ...)))))
       #:with (ctor-extra ...)
       (if star? #'((registered-names '(field-name ...) (list field ...))) #'())
       #:with (rebuild-extra ...) (if star? #'((registry*-fixed self)) #'())
       #:with rebuild-classify
       (if star?
           #'(classify-kept (registry*-fixed self) '(field-name ...) (list field ...))
           #'(classify '(field-name ...) (list field ...)))
       #:with export (contract-export stx #'name #'name? ctc pred)
       #'(begin
           fe.checker-def ... ...
           me.checker-def ... ...
           method-def ...
           (struct sid parent (field-spec ...)
             #:reflection-name reflect-name
             #:property prop:rebuilder
             (lambda (self child tensor) (rebuild self child tensor))
             moved-clause ...
             prop-clause ...)
           (define name? sid?)
           (define (rebuild self child tensor)
             (let ([field (rebuilt-value 'field-name (field-acc self) child tensor)]
                   ...)
               (let-values ([(params buffers children) rebuild-classify])
                 (check-names
                  'name
                  (sid forward-proc params buffers children (registry-mode self)
                       rebuild-extra ... field ...)))))
           (define forward-proc forward-expr)
           (define (name . formals)
             (let ([absent #f] ...)
               assign ...
               (let-values ([(params buffers children)
                             (classify '(field-name ...) (list field ...))])
                 (check-names
                  'name
                  (sid forward-proc params buffers children 'train
                       ctor-extra ... field ...)))))
           export)])))

(define-syntax (define-layer stx)
  (syntax-parse stx
    [(_ name:id (field:layer-field ...)
        (~alt (~optional (~seq #:init init:init-formals init-body:expr ...))
              (~optional (~seq #:reflection-name reflect:expr))
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))
              (~optional (~seq #:on-move moved-body:expr ...+))) ...
        #:forward fwd:entry-formals body:expr ...+)
     #:fail-when
     (unbare-field (attribute init) (attribute field.id) (attribute field.bare?))
     unbare-message
     #:with (formals (absent ...))
     (init-parts (attribute field.id) (attribute field.decl)
                 (attribute init.formals) (attribute init.id))
     (layer-expansion
      stx #f (attribute ctc) (attribute pred)
      #'(name formals (field.id ...) (~? (init-body ...) ()) (absent ...)
              (~? reflect 'name) (~? (moved-body ...) ()) ((fwd body ...)) () ()))]))

(define-syntax (define-layer* stx)
  (syntax-parse stx
    [(_ name:id (field:layer-field ...)
        (~alt (~optional (~seq #:init init:init-formals init-body:expr ...))
              (~optional (~seq #:reflection-name reflect:expr))
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))
              (~optional (~seq #:on-move moved-body:expr ...+))
              (~optional (~seq #:forward fwd:entry-formals fwd-body:expr ...+))
              (~seq #:method (m-name:id . m-formals:entry-formals) m-body:expr ...+)
              (~seq #:property prop:expr prop-val:expr)) ...)
     #:fail-when
     (unbare-field (attribute init) (attribute field.id) (attribute field.bare?))
     unbare-message
     #:with (formals (absent ...))
     (init-parts (attribute field.id) (attribute field.decl)
                 (attribute init.formals) (attribute init.id))
     (layer-expansion
      stx #t (attribute ctc) (attribute pred)
      #'(name formals (field.id ...) (~? (init-body ...) ()) (absent ...)
              (~? reflect 'name) (~? (moved-body ...) ())
              (~? ((fwd fwd-body ...)) ())
              ((m-name m-formals m-body ...) ...) ((prop prop-val) ...)))]))
