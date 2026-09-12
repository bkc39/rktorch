#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre)
         (only-in racket/contract/base
                  -> ->* ->i and/c any any/c cons/c contract-out contract?
                  flat-named-contract listof not/c or/c unsupplied-arg?)
         (only-in racket/generic define-generics)
         (only-in racket/list append-map check-duplicates remove-duplicates)
         (only-in racket/stxparam define-syntax-parameter syntax-parameterize)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign.rkt" prop:to tensor-dtype tensor?)
         (only-in (submod "../foreign.rkt" unsafe) to!)
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

(define (move-layer! m dev dtype)
  (when (and dtype (not (memq dtype '(float32 float64))))
    (raise-arguments-error 'to "a layer only moves to a floating-point dtype"
                           "dtype" dtype))
  (for ([t (in-list (append (parameters m) (buffers m)))])
    (define dt (and dtype (floating? t) dtype))
    (cond
      [(and dev dt) (to! t dev dt)]
      [dev (to! t dev)]
      [dt (to! t dt)]
      [else (void)]))
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

(struct registry (forward params buffers children [mode #:mutable])
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
      #:with formals #'((~@ f.decl ...) ...))))

(define-syntax (define-layer stx)
  (syntax-parse stx
    [(_ name:id (field:ctor-formal ...)
        (~alt (~optional (~seq #:init init:init-formals init-body:expr ...))
              (~optional (~seq #:reflection-name reflect:expr))
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))) ...
        #:forward (input:id ...) body:expr ...+)
     (define field-ids (syntax->list #'(field.id ...)))
     (for ([f (in-list field-ids)])
       (when (regexp-match? #rx"[.]" (symbol->string (syntax-e f)))
         (raise-syntax-error
          #f "a field name may not contain a dot; it is one state-dict segment"
          stx f)))
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
                    (filter (lambda (f)
                              (not (member f init-ids bound-identifier=?)))
                            (if init? field-ids '()))]
                   [(field-name ...)
                    (for/list ([f (in-list field-ids)])
                      (symbol->string (syntax-e f)))]
                   [(field-acc ...) (map accessor field-ids)]
                   [n-inputs (length (syntax->list #'(input ...)))])
       (with-syntax ([export (contract-export stx #'name #'name?
                                              (attribute ctc)
                                              (attribute pred))])
         #'(begin
             (struct sid registry (field.id ...)
               #:reflection-name reflect-name)
             (define name? sid?)
             (define (forward-proc self . inputs)
               (unless (= (length inputs) n-inputs)
                 (apply raise-arity-error 'name n-inputs inputs))
               (let ([field.id (field-acc self)] ...)
                 (syntax-parameterize
                     ([with-mode (with-mode-transformer #'self)])
                   (apply (lambda (input ...) body ...) inputs))))
             (define (name . formals)
               (let ([absent #f] ...)
                 init-body ...
                 (let-values ([(params buffers children)
                               (classify '(field-name ...)
                                         (list field.id ...))])
                   (check-names
                    'name
                    (sid forward-proc params buffers children 'train
                         field.id ...)))))
             export)))]))
