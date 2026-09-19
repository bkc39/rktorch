#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre
                     (only-in "../private/definer.rkt"
                              contract-export ctor-formal init-formals))
         (only-in racket/contract/base
                  -> ->* ->i and/c any any/c cons/c contract-out contract?
                  flat-named-contract listof not/c or/c unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in racket/list append-map check-duplicates remove-duplicates)
         (only-in racket/stxparam define-syntax-parameter syntax-parameterize)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt"
                  prop:to tensor-device tensor-dtype tensor? to)
         (only-in (submod "../foreign.rkt" unsafe) to!)
         (only-in "../foreign/autograd-ops.rkt" collect-at-forward-trough!)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "buffer.rkt" Buffer?)
         (only-in "parameter.rkt" Parameter?))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:layer
         layer?
         layer-forward ;; noqa
         layer-parameters ;; noqa
         layer-named-parameters ;; noqa
         layer-buffers ;; noqa
         layer-named-buffers ;; noqa
         layer-named-children ;; noqa
         layer-mode ;; noqa
         layer-set-mode! ;; noqa
         move-layer!
         call-at-forward-trough
         in-mode
         in-eval-mode
         with-mode
         define-layer)

(define-generics layer
  (layer-forward layer . inputs)
  (layer-parameters layer)
  (layer-named-parameters layer prefix)
  (layer-buffers layer)
  (layer-named-buffers layer prefix)
  (layer-named-children layer)
  (layer-mode layer)
  (layer-set-mode! layer mode)
  #:derive-property prop:to (lambda (m dev dtype) (move-layer! m dev dtype))
  #:fallbacks
  [(define (layer-parameters self) '()) ;; noqa
   (define (layer-named-parameters self prefix) '()) ;; noqa
   (define (layer-buffers self) '()) ;; noqa
   (define (layer-named-buffers self prefix) '()) ;; noqa
   (define (layer-named-children self) '()) ;; noqa
   (define (layer-mode self) 'train) ;; noqa
   (define (layer-set-mode! self mode) (void))]) ;; noqa

(define/contract-out mode/c contract?
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
(define (floating? t)
  (and (memq (tensor-dtype t) '(float32 float64)) #t))

;; What `#:on-move` reacts to: `to` is the identity when nothing changes, and
;; a device round trip returns to the placement it started from, so the pair
;; is read either side of one move rather than compared across several.
(define (placement-of m)
  (for/list ([t (in-list (append (parameters m) (buffers m)))])
    (cons (tensor-device t) (tensor-dtype t))))

(define (move-tensor! t dev dtype)
  (define dt (and dtype (floating? t) dtype))
  (cond
    [(and dev dt) (to! t dev dt)]
    [dev (to! t dev)]
    [dt (to! t dt)]
    [else (void)]))

;; A child moves through its own `to`, as nn.Module._apply recurses through
;; its children's, so a nested layer's #:on-move runs; what is left here is
;; whatever the children do not already own.
(define (own-tensors m)
  (define theirs (make-hasheq))
  (for* ([c (in-list (layer-named-children m))]
         [t (in-list (append (layer-parameters (cdr c))
                             (layer-buffers (cdr c))))])
    (hash-set! theirs t #t))
  (for/list ([t (in-list (append (parameters m) (buffers m)))]
             #:unless (hash-ref theirs t #f))
    t))

(define (move-layer! m dev dtype)
  (when (and dtype (not (memq dtype '(float32 float64))))
    (raise-arguments-error 'to "a layer only moves to a floating-point dtype"
                           "dtype" dtype))
  (for ([c (in-list (layer-named-children m))])
    (cond
      [(and dev dtype) (to (cdr c) dev dtype)]
      [dev (to (cdr c) dev)]
      [dtype (to (cdr c) dtype)]
      [else (void)]))
  (for ([t (in-list (own-tensors m))])
    (move-tensor! t dev dtype))
  m)

;; Depth-first, own params before children's, in declaration order —
;; PyTorch's parameters() order, which seeded-init parity relies on.
(define/contract-out (parameters m) ;; noqa
  (-> layer? (listof tensor?))
  (remove-duplicates (layer-parameters m) eq?))

(define/checked-out (named-parameters m [prefix ""]) ;; noqa
  (->* [layer?] [string?] (listof (cons/c string? tensor?)))
  (remove-duplicates (layer-named-parameters m prefix) eq? #:key cdr))

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

(define/contract-out (forward m . inputs) ;; noqa
  (-> layer? any/c ... any)
  (apply layer-forward m inputs))

(define/contract-out (train! m) ;; noqa
  (-> layer? layer?)
  (layer-set-mode! m 'train)
  m)

(define/contract-out (eval! m) ;; noqa
  (-> layer? layer?)
  (layer-set-mode! m 'eval)
  m)

(define/contract-out (set-mode! m mode) ;; noqa
  (-> layer? mode/c layer?)
  (layer-set-mode! m mode)
  m)

(define/contract-out (layer-training? m) ;; noqa
  (-> layer? boolean?)
  (training? (layer-mode m)))

(define (mode-snapshot m)
  (define seen (make-hasheq))
  (let walk ([m m])
    (cond
      [(hash-ref seen m #f) '()]
      [else
       (hash-set! seen m #t)
       (cons (cons m (layer-mode m))
             (append-map (lambda (c) (walk (cdr c)))
                         (layer-named-children m)))])))

(define (restore-modes! before)
  (for ([e (in-list before)] #:unless (registry? (car e)))
    (layer-set-mode! (car e) (cdr e)))
  (for ([e (in-list before)] #:when (registry? (car e)))
    (set-registry-mode! (car e) (cdr e))))

(define/contract-out (call-with-mode m mode thunk) ;; noqa
  (-> layer? mode/c (-> any) any)
  (define before (mode-snapshot m))
  (dynamic-wind (lambda () (layer-set-mode! m mode))
                thunk
                (lambda () (restore-modes! before))))

(define/contract-out (call-with-eval-mode m thunk) ;; noqa
  (-> layer? (-> any) any)
  (call-with-mode m 'eval thunk))

(define-syntax-parse-rule (in-mode m:expr mode:expr body:expr ...+)
  (call-with-mode m mode (lambda () body ...)))

(define-syntax-parse-rule (in-eval-mode m:expr body:expr ...+)
  (call-with-mode m 'eval (lambda () body ...)))

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

;; the mark tells a nested call from the outermost one, whose return is
;; where a no-grad loop's memory is at its lowest
(define (call-at-forward-trough forward)
  (cond
    [(continuation-mark-set-first #f layer-call-key) (forward)]
    [else
     (begin0 (with-continuation-mark layer-call-key #t (forward))
             (collect-at-forward-trough!))]))

(define (call-forward self inputs)
  (call-at-forward-trough
   (lambda () (apply (registry-forward self) self inputs))))

(struct registry (forward params buffers children [mode #:mutable])
  #:property prop:procedure
  (lambda (self . inputs) (call-forward self inputs))
  #:methods gen:layer
  [(define (layer-forward self . inputs)
     (call-forward self inputs))
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
   (define (layer-set-mode! self mode)
     (set-registry-mode! self mode)
     (for ([c (in-list (registry-children self))])
       (child-set-mode! c mode)))
   (define (layer-mode self)
     (registry-mode self))])

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
  #:reflection-name 'Fn)

(define (fn-forward self . inputs)
  (apply (Fn%-proc self) inputs))

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

(begin-for-syntax
  (define (check-field-names! stx fields)
    (for ([f (in-list fields)])
      (when (regexp-match? #rx"[.]" (symbol->string (syntax-e f)))
        (raise-syntax-error
         #f "a field name may not contain a dot; it is one state-dict segment"
         stx f))))

  (define (check-bare-fields! stx fields bare?s)
    (for ([f (in-list fields)] [bare? (in-list bare?s)])
      (unless bare?
        (raise-syntax-error
         #f
         "with #:init, a field is a bare identifier; defaults and keywords belong to the #:init formals"
         stx f))))

  ;; the fields an #:init leaves for its body to assign
  (define (unassigned-fields fields init-ids)
    (filter (lambda (f) (not (member f init-ids bound-identifier=?))) fields))

  (define (field-strings fields)
    (for/list ([f (in-list fields)]) (symbol->string (syntax-e f))))

  (define (accessor-ids struct-id fields)
    (for/list ([f (in-list fields)]) (format-id struct-id "~a-~a" struct-id f))))

(define-syntax (define-layer stx)
  (syntax-parse stx
    [(_ name:id (field:ctor-formal ...)
        (~alt (~optional (~seq #:init init:init-formals init-body:expr ...))
              (~optional (~seq #:reflection-name reflect:expr))
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))
              (~optional (~seq #:on-move moved-body:expr ...+))) ...
        #:forward (~or* (input:id ...) (input:id ... . restarg:id))
        body:expr ...+)
     #:do [(define fields (syntax->list #'(field.id ...)))
           (check-field-names! stx fields)
           (define init? (and (attribute init) #t))
           (when init?
             (check-bare-fields! stx fields (attribute field.bare?)))
           (define init-ids (if init? (syntax->list #'(init.id ...)) '()))
           (define struct-id (generate-temporary #'name))
           (define arity (length (syntax->list #'(input ...))))]
     #:with sid struct-id
     #:with sid? (format-id struct-id "~a?" struct-id)
     #:with name? (format-id #'name "~a?" #'name)
     #:with reflect-name (or (attribute reflect) #'(quote name))
     #:with formals (if init? #'init.formals #'((~@ field.decl ...) ...))
     #:with (assign ...) (if init? #'(init-body ...) #'())
     #:with (absent ...) (unassigned-fields (if init? fields '()) init-ids)
     #:with (field-name ...) (field-strings fields)
     #:with (field-acc ...) (accessor-ids struct-id fields)
     #:with n-inputs #`#,arity
     #:with enough? (if (attribute restarg) #'>= #'=)
     #:with expected (if (attribute restarg)
                         #`(arity-at-least #,arity)
                         #`#,arity)
     #:with forward-lambda (if (attribute restarg)
                               #'(lambda (input ... . restarg) body ...)
                               #'(lambda (input ...) body ...))
     ;; a move hook overrides the property gen:layer derives, so it is
     ;; attached only when #:on-move asked for it
     #:with (moved-defn ...)
     (if (attribute moved-body)
         #'((define (moved-proc self dev dtype)
              (define before (placement-of self))
              (begin0 (move-layer! self dev dtype)
                      (unless (equal? before (placement-of self))
                        (let ([field.id (field-acc self)] ...)
                          moved-body ...)))))
         #'())
     #:with (moved-clause ...)
     (if (attribute moved-body) #'(#:property prop:to moved-proc) #'())
     #:with export (contract-export stx #'name #'name?
                                    (attribute ctc) (attribute pred))
     #'(begin
         moved-defn ...
         (struct sid registry (field.id ...)
           #:reflection-name reflect-name
           moved-clause ...)
         (define name? sid?)
         (define (forward-proc self . inputs)
           (unless (enough? (length inputs) n-inputs)
             (apply raise-arity-error 'name expected inputs))
           (let ([field.id (field-acc self)] ...)
             (syntax-parameterize ([with-mode (with-mode-transformer #'self)])
               (apply forward-lambda inputs))))
         (define (name . formals)
           (let ([absent #f] ...)
             assign ...
             (let-values ([(params buffers children)
                           (classify '(field-name ...) (list field.id ...))])
               (check-names
                'name
                (sid forward-proc params buffers children 'train
                     field.id ...)))))
         export)]))
