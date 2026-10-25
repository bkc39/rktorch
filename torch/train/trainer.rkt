#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/list check-duplicates)
                     (only-in racket/match match)
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre
                              ...+ ~alt ~and ~datum ~literal ~optional ~or* ~seq ~var
                              attribute boolean define-splicing-syntax-class
                              define-syntax-class expr id pattern syntax-parse)
                     (only-in "../private/definer.rkt"
                              contract-export ctor-formal forward-formal
                              forward-keyword init-formals)
                     (only-in "private/hooks.rkt" hook-arity-problem hook-for-side))
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
    #:attributes (id unbare)
    (pattern f:ctor-formal
      #:fail-when
      (and (regexp-match? #rx"[.]" (symbol->string (syntax-e #'f.id))) #'f.id)
      "a field name may not contain a dot; it is one state-dict segment"
      #:with id #'f.id
      #:attr unbare (and (not (attribute f.bare?)) #'f.id)))

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

  (define-syntax-class any-body-form
    #:description "a define-trainer or define-callback body form"
    #:attributes (owner)
    (pattern ((~literal define-init) :init-formals _:expr ...+)
      #:attr owner 'both)
    (pattern ((~literal define-forward) :forward-formals _:expr ...+)
      #:attr owner 'trainer)
    (pattern ((~or* (~literal define-optimizers) (~literal define-callbacks)) _:expr ...+)
      #:attr owner 'trainer)
    (pattern ((~literal define-hparams) _:id ...+)
      #:attr owner 'trainer)
    (pattern ((~or* (~literal define-training-step) (~literal define-validation-step)
                    (~literal define-test-step) (~literal define-predict-step))
              :step-formals _:expr ...+)
      #:attr owner 'trainer)
    (pattern ((~literal define-hook) (_:id . _:hook-params) _:expr ...+)
      #:attr owner 'both)
    (pattern ((~literal define-state-key) _:expr)
      #:attr owner 'callback))

  (define (form-kind form)
    (match (syntax-e form)
      [(cons head _) (match (identifier-binding head) [(list* _ name _) name])]))

  (define (forms-of kind forms)
    (filter (lambda (f) (eq? (form-kind f) kind)) forms))

  (define-syntax-class (body-form side)
    (pattern form:any-body-form
      #:fail-unless (memq (attribute form.owner) (list side 'both))
      (format "~a is a define-~a form" (form-kind #'form) (attribute form.owner))))

  (define-syntax-class (hook-name side)
    #:attributes (hook)
    (pattern id:id
      #:do [(define-values (h problem) (hook-for-side (syntax-e #'id) side))]
      #:fail-when problem problem
      #:attr hook h))

  (define-syntax-class (hook-header side)
    #:attributes (id)
    (pattern ((~var id (hook-name side)) . formals:hook-params)
      #:do [(define problem (hook-arity-problem (attribute id.hook)
                                                (attribute formals.count)
                                                (attribute formals.rest?)))]
      #:fail-when (and problem #'id) problem))

  (define-syntax-class (field-of fields trainer)
    (pattern h:id
      #:fail-unless (memq (syntax-e #'h) fields)
      (format "~a is not a field of ~a" (syntax-e #'h) trainer)))

  (define ((definer who side) stx)
    (syntax-parse stx
      [(_ name:id (field:definer-field ...) (~var opt (definer-options side))
          (~var form (body-form side)) ...)
       #:do [(define forms (attribute form))
             (define kinds (map form-kind forms))
             (define repeated (check-duplicates (remq* '(define-hook) kinds)))
             (define unbare
               (and (memq 'define-init kinds) (findf values (attribute field.unbare))))]
       #:fail-when repeated (format "~a appears twice" repeated)
       #:fail-when unbare
       (string-append "with define-init, a field is a bare identifier;"
                      " defaults and keywords belong to the define-init formals")
       #:do [(syntax-parse #`(#,@(forms-of 'define-hook forms))
               #:context (list 'define-hook stx)
               [((_ (~var header (hook-header side)) . _) ...)
                #:do [(define twice
                        (check-duplicates (attribute header.id) eq? #:key syntax-e))]
                #:fail-when twice (format "~a is defined twice" (syntax-e twice))
                (void)])
             (define fields (map syntax-e (attribute field.id)))
             (syntax-parse #`(#,@(forms-of 'define-hparams forms))
               #:context (list 'define-hparams stx)
               [((_ (~var _ (field-of fields (syntax-e #'name))) ...) ...) (void)])]
       #:with name? (format-id #'name "~a?" #'name)
       #:with export (contract-export stx #'name #'name?
                                      (attribute opt.ctc) (attribute opt.pred))
       #`(begin
           (define (name . _) (not-implemented '#,who 'T1))
           (define (name? _) #f)
           export)])))

(define-syntax define-trainer (definer 'define-trainer 'trainer))
(define-syntax define-callback (definer 'define-callback 'callback))
