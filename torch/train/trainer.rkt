#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/list check-duplicates)
                     (only-in racket/match match)
                     (only-in racket/syntax generate-temporary)
                     (only-in syntax/parse/pre
                              ...+ ~alt ~and ~datum ~literal ~optional ~or* ~seq ~var
                              attribute boolean define-splicing-syntax-class
                              define-syntax-class expr id pattern syntax-parse)
                     (only-in "../private/definer.rkt"
                              field-of forward-formal forward-keyword init-formals
                              layer-fields)
                     (only-in "private/hooks.rkt"
                              hook-arity-problem hook-for-side hook-log hook-named))
         (only-in racket/contract/base -> any/c contract-out)
         (only-in racket/stxparam define-syntax-parameter syntax-parameterize)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in (submod "../nn/layer.rkt" private) define-layer*)
         (only-in "logging.rkt" run-log!)
         (only-in "private/checks.rkt"
                  checked-state-key configured-callbacks configured-optimizers
                  readable-arguments training-step-result)
         (only-in "private/types.rkt"
                  callback-property callback? prop:callback prop:trainer trainer-info
                  trainer?))

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
            (keyword-apply run-log! kws kw-args #,run-id rest)))]))

  (define ((cannot-log hook) stx)
    (raise-syntax-error 'log! (format "~a cannot log" hook) stx)))

(define-syntax-parse-rule (define-body-forms name:id ...)
  (begin (provide name ...) (define-syntax name outside-definer) ...))

(define-syntax-parse-rule (define-hook-parameters name:id ...)
  (begin (provide name ...) (define-syntax-parameter name outside-hook) ...))

(define-body-forms
  define-callbacks define-forward define-hook define-hparams define-init
  define-optimizers define-predict-step define-state-key define-test-step
  define-training-step define-validation-step)

(define-hook-parameters log! run)

(define-syntax (with-hook-context stx)
  (syntax-parse stx
    [(_ hook:id r:id body:expr ...+)
     #:with log-transformer
     (if (hook-log (hook-named (syntax-e #'hook)))
         #'(hook-log! (quote-syntax r))
         #'(cannot-log 'hook))
     #'(syntax-parameterize ([run (make-rename-transformer (quote-syntax r))]
                             [log! log-transformer])
         body ...)]))

(begin-for-syntax
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
    #:attributes (ctc pred automatic kind)
    (pattern (~seq (~alt (~optional (~seq #:contract ctc:expr))
                         (~optional (~seq #:predicate pred:id))
                         (~optional (~seq (~and auto #:automatic-optimization?)
                                          automatic:boolean))
                         (~optional (~seq (~and kind-kw #:kind) kind:callback-kind)))
                   ...)
      #:fail-when (and (eq? side 'callback) (attribute auto))
      "#:automatic-optimization? is a define-trainer option"
      #:fail-when (and (eq? side 'trainer) (attribute kind-kw))
      "#:kind is a define-callback option"))

  (define (method r name formals body)
    #`(#:method #:hidden-arguments 1 (#,name #,r . #,formals)
       (with-hook-context #,name #,r #,@body)))

  (define-syntax-class (any-body-form automatic?)
    #:description "a define-trainer or define-callback body form"
    #:attributes (kind owner (clause 1))
    (pattern ((~literal define-init) :init-formals _:expr ...+)
      #:attr kind 'define-init
      #:attr owner 'both
      #:with (clause ...) #'())
    (pattern ((~literal define-forward) formals:forward-formals body:expr ...+)
      #:attr kind 'define-forward
      #:attr owner 'trainer
      #:with (clause ...) #'(#:forward formals body ...))
    (pattern ((~literal define-optimizers) body:expr ...+)
      #:attr kind 'define-optimizers
      #:attr owner 'trainer
      #:with (clause ...)
      (method (generate-temporary 'run) #'configure-optimizers #'()
              #`((configured-optimizers #,automatic?
                                        (call-with-values (lambda () body ...) list)))))
    (pattern ((~literal define-callbacks) body:expr ...+)
      #:attr kind 'define-callbacks
      #:attr owner 'trainer
      #:with (clause ...)
      (method (generate-temporary 'run) #'configure-callbacks #'()
              #'((configured-callbacks (call-with-values (lambda () body ...) list)))))
    (pattern ((~literal define-hparams) _:id ...+)
      #:attr kind 'define-hparams
      #:attr owner 'trainer
      #:with (clause ...) #'())
    (pattern ((~literal define-training-step) formals:step-formals body:expr ...+)
      #:attr kind 'define-training-step
      #:attr owner 'trainer
      #:with (clause ...)
      (method (generate-temporary 'run) #'training-step #'formals
              #'((call-with-values (lambda () body ...)
                                   (lambda results (training-step-result results))))))
    (pattern ((~literal define-validation-step) formals:step-formals body:expr ...+)
      #:attr kind 'define-validation-step
      #:attr owner 'trainer
      #:with (clause ...)
      (method (generate-temporary 'run) #'validation-step #'formals #'(body ...)))
    (pattern ((~literal define-test-step) formals:step-formals body:expr ...+)
      #:attr kind 'define-test-step
      #:attr owner 'trainer
      #:with (clause ...) (method (generate-temporary 'run) #'test-step #'formals #'(body ...)))
    (pattern ((~literal define-predict-step) formals:step-formals body:expr ...+)
      #:attr kind 'define-predict-step
      #:attr owner 'trainer
      #:with (clause ...)
      (method (generate-temporary 'run) #'predict-step #'formals #'(body ...)))
    (pattern ((~literal define-hook) (hook:id . formals:hook-params) body:expr ...+)
      #:attr kind 'define-hook
      #:attr owner 'both
      #:with (clause ...) (method (generate-temporary 'run) #'hook #'formals #'(body ...)))
    (pattern ((~literal define-state-key) key:expr)
      #:attr kind 'define-state-key
      #:attr owner 'callback
      #:with (clause ...) #'(#:method (state-key) (checked-state-key key))))

  (define (forms-of kind forms kinds)
    (for/list ([f (in-list forms)] [k (in-list kinds)] #:when (eq? k kind)) f))

  (define-syntax-class (body-form side automatic?)
    #:attributes (kind (clause 1))
    (pattern (~var form (any-body-form automatic?))
      #:fail-unless (memq (attribute form.owner) (list side 'both))
      (format "~a is a define-~a form" (attribute form.kind) (attribute form.owner))
      #:attr kind (attribute form.kind)
      #:with (clause ...) #'(form.clause ...)))

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

  (define (capture args ids)
    #`(set! #,args (let ([given (readable-arguments '#,ids (list #,@ids))])
                     (lambda () given))))

  (define (init-clauses init-form args field-decls)
    (syntax-parse (or init-form #`(define-init #,@field-decls))
      [(_ formals:init-formals body ...)
       #`(#:init formals #,(capture args (attribute formals.id)) body ...)]))

  (define (callback-init-clauses init-form)
    (syntax-parse (or init-form #'(_))
      [(_ formals body ...) #'(#:init formals body ...)]
      [(_) #'()]))

  (define ((definer who side) stx)
    (syntax-parse stx
      [(_ name:id fields (~var opt (definer-options side))
          (~var form (body-form side (if (attribute opt.automatic)
                                         (syntax-e #'opt.automatic)
                                         #t)))
          ...)
       #:do [(define forms (attribute form))
             (define kinds (attribute form.kind))
             (define repeated (check-duplicates (remq* '(define-hook) kinds)))]
       #:fail-when repeated (format "~a appears twice" repeated)
       #:with (~var f (layer-fields (and (memq 'define-init kinds) "define-init"))) #'fields
       #:do [(syntax-parse #`(#,@(forms-of 'define-hook forms kinds))
               #:context (list 'define-hook stx)
               [((_ (~var header (hook-header side)) . _) ...)
                #:do [(define twice
                        (check-duplicates (attribute header.id) eq? #:key syntax-e))]
                #:fail-when twice (format "~a is defined twice" (syntax-e twice))
                (void)])
             (define field-names (map syntax-e (attribute f.id)))
             (define hparams
               (syntax-parse #`(#,@(forms-of 'define-hparams forms kinds))
                 #:context (list 'define-hparams stx)
                 [((_ (~var h (field-of field-names (syntax-e #'name))) ...) ...)
                  (apply append (attribute h))]))
             (define init-form (findf values (forms-of 'define-init forms kinds)))
             (define automatic? (if (attribute opt.automatic) (syntax-e #'opt.automatic) #t))]
       #:with (hparam ...) hparams
       (match side
         ['trainer
          (define args (generate-temporary 'arguments))
          #`(define-layer* name (f.id ... #,args)
              #,@(init-clauses init-form args #'(((~@ f.decl ...) ...)))
              (~? (~@ #:contract opt.ctc))
              (~? (~@ #:predicate opt.pred))
              #:property prop:trainer (trainer-info '#,kinds #,automatic? '(hparam ...))
              #:method (hyperparameters)
              (hash-set* (#,args) (~@ 'hparam hparam) ...)
              form.clause ... ...)]
         ['callback
          #`(define-layer* name fields
              #,@(callback-init-clauses init-form)
              (~? (~@ #:contract opt.ctc))
              (~? (~@ #:predicate opt.pred))
              #:property prop:callback
              (callback-property '#,(if (attribute opt.kind) (syntax-e #'opt.kind) 'plain)
                                 (gensym 'name))
              #,@(if (memq 'define-state-key kinds)
                     #'()
                     #'(#:method (state-key) (symbol->string 'name)))
              form.clause ... ...)])])))

(define-syntax define-trainer (definer 'define-trainer 'trainer))
(define-syntax define-callback (definer 'define-callback 'callback))
