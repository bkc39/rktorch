#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/list check-duplicates filter-not)
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre
                              ...+ ~alt ~and ~datum ~literal ~optional ~or* ~seq ~var
                              attribute boolean define-splicing-syntax-class
                              define-syntax-class expr id pattern syntax-parse)
                     (only-in "../private/definer.rkt"
                              contract-export ctor-formal forward-formal
                              forward-keyword init-formals)
                     (only-in "private/hooks.rkt"
                              callback-hook? closest-hook-name
                              definable-hook-named hook-batch? hook-formals
                              trainer-hook?))
         (only-in racket/contract/base -> any/c contract-out)
         (only-in racket/stxparam define-syntax-parameter)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "logging.rkt" run-log!)
         (only-in "private/types.rkt" callback? not-implemented trainer?))

(provide define-callback
         define-trainer
         (contract-out [callback? (-> any/c boolean?)]
                       [trainer? (-> any/c boolean?)]))

(module+ private
  (provide (for-syntax hook-log!)))

(begin-for-syntax
  (define (outside-definer stx)
    (raise-syntax-error
     #f "only allowed directly inside define-trainer or define-callback" stx))

  (define (outside-hook stx)
    (raise-syntax-error
     #f "only allowed inside a trainer or callback hook body" stx))

  (define ((hook-log! run-id) stx)
    (syntax-parse stx
      [(_ arg ...) #`(run-log! #,run-id arg ...)]
      [_:id
       #`(make-keyword-procedure
          (lambda (kws kw-args . rest)
            (keyword-apply run-log! kws kw-args #,run-id rest)))])))

(define-syntax-parse-rule (define-body-forms name:id ...)
  (begin (provide name ...) (define-syntax name outside-definer) ...))

(define-syntax-parse-rule (define-hook-parameters name:id ...)
  (begin (provide name ...) (define-syntax-parameter name outside-hook) ...))

(define-body-forms
  define-callbacks define-forward define-hook define-hparams define-init
  define-optimizers define-predict-step define-state-key define-test-step
  define-training-step define-validation-step)

(define-hook-parameters log! run)

(begin-for-syntax
  (define-splicing-syntax-class definer-field
    #:description "a field"
    #:attributes (id bare?)
    (pattern f:ctor-formal
      #:fail-when
      (and (regexp-match? #rx"[.]" (symbol->string (syntax-e #'f.id))) #'f.id)
      "a field name may not contain a dot; it is one state-dict segment"
      #:with id #'f.id
      #:attr bare? (attribute f.bare?)))

  (define-syntax-class forward-formals
    #:description "forward formals"
    (pattern ((~alt _:forward-formal _:forward-keyword) ... . (~or* () _:id))))

  (define-syntax-class step-formals
    #:description "step formals: (input ...) or (input ... . rest-id)"
    (pattern (_:forward-formal ... . (~or* () _:id))))

  (define-syntax-class hook-params
    #:description "hook formals: (id ...) or (id ... . rest-id)"
    #:attributes (count rest?)
    (pattern (x:id ... . (~or* () r:id))
      #:attr count (length (attribute x))
      #:attr rest? (and (attribute r) #t)))

  (define-syntax-class callback-kind
    #:description "callback kind (plain, monitor or checkpoint)"
    #:opaque
    (pattern (~or* (~datum plain) (~datum monitor) (~datum checkpoint))))

  (define-splicing-syntax-class (definer-options side)
    #:attributes (ctc pred)
    (pattern (~seq (~alt (~optional (~seq #:contract ctc:expr))
                         (~optional (~seq #:predicate pred:id))
                         (~optional (~seq (~and auto #:automatic-optimization?)
                                          _:boolean))
                         (~optional (~seq (~and kind #:kind) _:callback-kind)))
                   ...)
      #:fail-when (and (eq? side 'callback) (attribute auto))
      "#:automatic-optimization? is a define-trainer option"
      #:fail-when (and (eq? side 'trainer) (attribute kind))
      "#:kind is a define-callback option"))

  (define-syntax-class step-head
    #:attributes (kind)
    (pattern (~literal define-training-step) #:attr kind 'define-training-step)
    (pattern (~literal define-validation-step) #:attr kind 'define-validation-step)
    (pattern (~literal define-test-step) #:attr kind 'define-test-step)
    (pattern (~literal define-predict-step) #:attr kind 'define-predict-step))

  (define-syntax-class any-body-form
    #:description "a define-trainer or define-callback body form"
    #:attributes (kind)
    (pattern ((~literal define-init) :init-formals _:expr ...+)
      #:attr kind 'define-init)
    (pattern ((~literal define-forward) :forward-formals _:expr ...+)
      #:attr kind 'define-forward)
    (pattern ((~literal define-optimizers) _:expr ...+)
      #:attr kind 'define-optimizers)
    (pattern ((~literal define-callbacks) _:expr ...+)
      #:attr kind 'define-callbacks)
    (pattern ((~literal define-hparams) _:id ...+)
      #:attr kind 'define-hparams)
    (pattern (head:step-head :step-formals _:expr ...+)
      #:attr kind (attribute head.kind))
    (pattern ((~literal define-hook) (_:id . _:hook-params) _:expr ...+)
      #:attr kind 'define-hook)
    (pattern ((~literal define-state-key) _:expr)
      #:attr kind 'define-state-key))

  (define callback-forms '(define-init define-state-key define-hook))

  (define-syntax-class (body-form side)
    #:attributes (kind)
    (pattern form:any-body-form
      #:attr kind (attribute form.kind)
      #:fail-when (and (eq? side 'callback)
                       (not (memq (attribute kind) callback-forms))
                       #'form)
      (format "~a is a define-trainer form" (attribute kind))
      #:fail-when (and (eq? side 'trainer)
                       (eq? (attribute kind) 'define-state-key)
                       #'form)
      (format "~a is a define-callback form" (attribute kind))))

  (define (arguments formals)
    (case (length formals)
      [(0) "no arguments"]
      [(1) (format "1 argument ~a" formals)]
      [else (format "~a arguments ~a" (length formals) formals)]))

  (define-syntax-class (hook-header side)
    #:attributes (id)
    (pattern (id:id . formals:hook-params)
      #:do [(define name (syntax-e #'id))
            (define h (definable-hook-named name))]
      #:fail-when (and (not h) #'id)
      (format "unknown hook ~a; closest: ~a" name (closest-hook-name name))
      #:fail-when (and (eq? side 'callback) (not (callback-hook? h)) #'id)
      (format "~a is a trainer hook; define-callback cannot define it" name)
      #:fail-when (and (eq? side 'trainer) (not (trainer-hook? h)) #'id)
      (format "~a is a callback hook; define-trainer cannot define it" name)
      #:do [(define expected (hook-formals h))
            (define n (length expected))]
      #:fail-when (and (hook-batch? h) (< (attribute formals.count) n) #'id)
      (format "~a takes ~a then the batch values" name (arguments expected))
      #:fail-when (and (not (hook-batch? h))
                       (or (attribute formals.rest?)
                           (not (= (attribute formals.count) n)))
                       #'id)
      (format "~a takes ~a" name (arguments expected))))

  (define-syntax-class distinct-hooks
    (pattern (id:id ...)
      #:do [(define twice (check-duplicates (attribute id) eq? #:key syntax-e))]
      #:fail-when twice (format "~a is defined twice" (and twice (syntax-e twice)))))

  (define-syntax-class (hparam-fields fields trainer)
    (pattern (_ h:id ...+)
      #:do [(define stray
              (for/first ([h (in-list (attribute h))]
                          #:unless (memq (syntax-e h) fields))
                h))]
      #:fail-when stray
      (format "~a is not a field of ~a" (and stray (syntax-e stray)) trainer)))

  (define ((forms-of kind) forms kinds)
    (for/list ([f (in-list forms)] [k (in-list kinds)] #:when (eq? k kind))
      f))

  (define hook-forms (forms-of 'define-hook))
  (define hparam-forms (forms-of 'define-hparams))

  (define ((definer who side) stx)
    (syntax-parse stx
      [(_ name:id (field:definer-field ...) (~var opt (definer-options side))
          (~var form (body-form side)) ...)
       #:do [(define forms (attribute form))
             (define kinds (attribute form.kind))
             (define repeated
               (check-duplicates
                (filter-not (lambda (k) (eq? k 'define-hook)) kinds)))
             (define unbare
               (and (memq 'define-init kinds)
                    (for/first ([f (in-list (attribute field.id))]
                                [bare? (in-list (attribute field.bare?))]
                                #:unless bare?)
                      f)))]
       #:fail-when repeated (format "~a appears twice" repeated)
       #:fail-when unbare
       (string-append "with define-init, a field is a bare identifier;"
                      " defaults and keywords belong to the define-init formals")
       #:with (hook-id ...)
       (for/list ([f (in-list (hook-forms forms kinds))])
         (syntax-parse f
           #:context (list 'define-hook f)
           [(_ (~var header (hook-header side)) . _) #'header.id]))
       #:do [(syntax-parse #'(hook-id ...)
               #:context (list 'define-hook stx)
               [_:distinct-hooks (void)])
             (define fields (map syntax-e (attribute field.id)))
             (for ([f (in-list (hparam-forms forms kinds))])
               (syntax-parse f
                 #:context (list 'define-hparams stx)
                 [(~var _ (hparam-fields fields (syntax-e #'name))) (void)]))]
       #:with name? (format-id #'name "~a?" #'name)
       #:with export (contract-export stx #'name #'name?
                                      (attribute opt.ctc) (attribute opt.pred))
       #`(begin
           (define (name . _) (not-implemented '#,who 'T1))
           (define (name? _) #f)
           export)])))

(define-syntax define-trainer (definer 'define-trainer 'trainer))
(define-syntax define-callback (definer 'define-callback 'callback))
