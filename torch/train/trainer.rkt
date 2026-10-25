#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/list check-duplicates)
                     (only-in racket/syntax format-id)
                     ;; whole-module on purpose: the syntax classes need
                     ;; bindings only-in would strip
                     syntax/parse/pre
                     (only-in "../private/definer.rkt"
                              contract-export ctor-formal forward-formal
                              forward-keyword init-formals)
                     (only-in "private/hooks.rkt"
                              callback-hook? closest-hook-name
                              definable-hook-named hook-batch? hook-formals
                              trainer-hook?))
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "private/types.rkt" not-implemented))

(provide define-callback
         define-callbacks
         define-forward
         define-hook
         define-hparams
         define-init
         define-optimizers
         define-predict-step
         define-state-key
         define-test-step
         define-trainer
         define-training-step
         define-validation-step)

(begin-for-syntax
  (define (outside-definer stx)
    (raise-syntax-error
     #f "only allowed directly inside define-trainer or define-callback" stx)))

(define-syntax-parse-rule (define-body-forms name:id ...)
  (begin (define-syntax name outside-definer) ...))

(define-body-forms
  define-callbacks define-forward define-hook define-hparams define-init
  define-optimizers define-predict-step define-state-key define-test-step
  define-training-step define-validation-step)

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
    (pattern ((~alt input:forward-formal kw:forward-keyword) ...))
    (pattern ((~alt input:forward-formal kw:forward-keyword) ... . rest:id)))

  (define-syntax-class step-formals
    #:description "step formals: (input ...) or (input ... . rest-id)"
    (pattern (input:forward-formal ...))
    (pattern (input:forward-formal ... . rest:id)))

  (define-syntax-class hook-params
    #:description "hook formals: (id ...) or (id ... . rest-id)"
    #:attributes (count rest?)
    (pattern (x:id ...)
      #:attr count (length (syntax->list #'(x ...)))
      #:attr rest? #f)
    (pattern (x:id ... . r:id)
      #:attr count (length (syntax->list #'(x ...)))
      #:attr rest? #t))

  (define-syntax-class callback-kind
    #:description "callback kind (plain, monitor or checkpoint)"
    #:opaque
    (pattern (~or* (~datum plain) (~datum monitor) (~datum checkpoint))))

  (define (form-kind head)
    (for/first ([k (in-list (list #'define-optimizers #'define-callbacks
                                  #'define-training-step
                                  #'define-validation-step #'define-test-step
                                  #'define-predict-step))]
                #:when (free-identifier=? head k))
      (syntax-e k)))

  (define-syntax-class body-form
    #:description "a define-trainer or define-callback body form"
    #:attributes (kind)
    (pattern ((~literal define-init) :init-formals _:expr ...+)
      #:attr kind 'define-init)
    (pattern ((~literal define-forward) :forward-formals _:expr ...+)
      #:attr kind 'define-forward)
    (pattern ((~and head (~or* (~literal define-optimizers)
                               (~literal define-callbacks)))
              _:expr ...+)
      #:attr kind (form-kind #'head))
    (pattern ((~literal define-hparams) _:id ...+)
      #:attr kind 'define-hparams)
    (pattern ((~and head (~or* (~literal define-training-step)
                               (~literal define-validation-step)
                               (~literal define-test-step)
                               (~literal define-predict-step)))
              :step-formals _:expr ...+)
      #:attr kind (form-kind #'head))
    (pattern ((~literal define-hook) (_:id . _:hook-params) _:expr ...+)
      #:attr kind 'define-hook)
    (pattern ((~literal define-state-key) _:expr)
      #:attr kind 'define-state-key))

  (define callback-forms '(define-init define-state-key define-hook))

  (define (arguments n)
    (case n
      [(0) "no arguments"]
      [(1) "1 argument"]
      [else (format "~a arguments" n)]))

  (define (check-hook! side form)
    (syntax-parse form
      [(_ (id:id . formals:hook-params) . _)
       (define name (syntax-e #'id))
       (define h (definable-hook-named name))
       (define expected (and h (hook-formals h)))
       (define n (and h (length expected)))
       (define count (attribute formals.count))
       (define (fail fmt . args)
         (raise-syntax-error 'define-hook (apply format fmt args) form #'id))
       (cond
         [(not h)
          (fail "unknown hook ~a; closest: ~a" name (closest-hook-name name))]
         [(and (eq? side 'callback) (not (callback-hook? h)))
          (fail "~a is a trainer hook; define-callback cannot define it" name)]
         [(and (eq? side 'trainer) (not (trainer-hook? h)))
          (fail "~a is a callback hook; define-trainer cannot define it" name)]
         [(and (hook-batch? h) (< count n))
          (fail "~a takes ~a ~a then the batch values" name (arguments n)
                expected)]
         [(and (not (hook-batch? h)) (or (attribute formals.rest?) (not (= count n))))
          (fail "~a takes ~a ~a" name (arguments n) expected)]
         [else #'id])]))

  (define (hparam-ids form)
    (syntax-parse form
      [(_ id:id ...) (syntax->list #'(id ...))]))

  (define (check-body! who side stx name fields bare? forms kinds)
    (define (fail-at sub fmt . args)
      (raise-syntax-error who (apply format fmt args) stx sub))
    (for ([form (in-list forms)] [kind (in-list kinds)])
      (cond
        [(and (eq? side 'callback) (not (memq kind callback-forms)))
         (fail-at form "~a is a define-trainer form" kind)]
        [(and (eq? side 'trainer) (eq? kind 'define-state-key))
         (fail-at form "~a is a define-callback form" kind)]
        [else (void)]))
    (define repeated
      (check-duplicates (filter (lambda (k) (not (eq? k 'define-hook))) kinds)))
    (when repeated
      (fail-at #f "~a appears twice" repeated))
    (define unbare
      (for/first ([f (in-list fields)] [b? (in-list bare?)]
                  #:when (and (memq 'define-init kinds) (not b?)))
        f))
    (when unbare
      (fail-at unbare
               (string-append
                "with define-init, a field is a bare identifier;"
                " defaults and keywords belong to the define-init formals")))
    (define hook-ids
      (for/list ([form (in-list forms)] [kind (in-list kinds)]
                 #:when (eq? kind 'define-hook))
        (check-hook! side form)))
    (define hook-repeat (check-duplicates hook-ids eq? #:key syntax-e))
    (when hook-repeat
      (raise-syntax-error
       'define-hook (format "~a is defined twice" (syntax-e hook-repeat))
       stx hook-repeat))
    (for* ([form (in-list forms)]
           #:when (free-identifier=? (car (syntax-e form)) #'define-hparams)
           [h (in-list (hparam-ids form))]
           #:unless (memq (syntax-e h) (map syntax-e fields)))
      (raise-syntax-error
       'define-hparams
       (format "~a is not a field of ~a" (syntax-e h) (syntax-e name))
       stx h))))

(define-syntax (define-trainer stx)
  (syntax-parse stx
    [(_ name:id (field:definer-field ...)
        (~alt (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))
              (~optional (~seq #:reflection-name _:expr))
              (~optional (~seq #:automatic-optimization? (~or* #t #f))))
        ...
        form:body-form ...)
     #:do [(check-body! 'define-trainer 'trainer stx #'name
                        (syntax->list #'(field.id ...)) (attribute field.bare?)
                        (syntax->list #'(form ...)) (attribute form.kind))]
     #:with name? (format-id #'name "~a?" #'name)
     #:with export (contract-export stx #'name #'name?
                                    (attribute ctc) (attribute pred))
     #'(begin
         (define (name . _) (not-implemented 'define-trainer 'T1))
         (define (name? _) #f)
         export)]))

(define-syntax (define-callback stx)
  (syntax-parse stx
    [(_ name:id (field:definer-field ...)
        (~alt (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))
              (~optional (~seq #:reflection-name _:expr))
              (~optional (~seq #:kind _:callback-kind)))
        ...
        form:body-form ...)
     #:do [(check-body! 'define-callback 'callback stx #'name
                        (syntax->list #'(field.id ...)) (attribute field.bare?)
                        (syntax->list #'(form ...)) (attribute form.kind))]
     #:with name? (format-id #'name "~a?" #'name)
     #:with export (contract-export stx #'name #'name?
                                    (attribute ctc) (attribute pred))
     #'(begin
         (define (name . _) (not-implemented 'define-callback 'T1))
         (define (name? _) #f)
         export)]))
