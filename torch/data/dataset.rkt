#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre
                     (only-in "../private/definer.rkt"
                              contract-export ctor-formal init-formals))
         (only-in racket/contract/base
                  -> ->i any any/c contract-out contract? flat-named-contract
                  non-empty-listof or/c)
         (only-in racket/generic define-generics)
         (only-in racket/list first)
         (only-in "../foreign.rkt"
                  device? dtype shape stack tensor-device tensor->list tensor?)
         (only-in "../private/contract.rkt" define/contract-out))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:dataset
         define-dataset
         (contract-out
          [dataset? (-> any/c boolean?)]
          [dataset-length (-> dataset? exact-nonnegative-integer?)]
          [dataset-ref (->i ([ds dataset?] [i (ds) (index-of/c ds)]) any)]
          [dataset-batch (->i ([ds dataset?]
                               [indices (ds) (indices-of/c ds)]
                               [collate collate/c])
                              any)]
          [dataset-device (-> dataset? (or/c device? #f))]))

(define index-tensor/c
  (flat-named-contract
   'non-empty-int64-vector
   (lambda (v)
     (and (tensor? v)
          (eq? (dtype v) 'int64)
          (= 1 (length (shape v)))
          (positive? (car (shape v)))))))
(define/contract-out indices/c contract? ;; noqa
  (or/c (non-empty-listof exact-nonnegative-integer?) index-tensor/c))
(define/contract-out collate/c contract? ;; noqa
  (-> (non-empty-listof list?) any))

(define (index-of/c ds)
  (define n (dataset-length ds))
  (flat-named-contract 'index-below-length
                       (lambda (i) (and (exact-nonnegative-integer? i) (< i n)))))

;; a list is checked against the length; a tensor's elements are not read
(define (indices-of/c ds)
  (define below? (index-of/c ds))
  (or/c (flat-named-contract
         'indices-below-length
         (lambda (v) (and (list? v) (pair? v) (andmap below? v))))
        index-tensor/c))

(define-generics dataset
  (dataset-length dataset)
  (dataset-ref dataset i)
  (dataset-batch dataset indices collate)
  (dataset-device dataset)
  #:fallbacks
  [(define (dataset-batch self indices collate) ;; noqa
     (batch-by-ref self indices collate))
   (define (dataset-device self) #f)]) ;; noqa

(define (batch-by-ref ds indices collate) ;; noqa
  (collate (for/list ([i (in-list (indices->list indices))])
             (call-with-values (lambda () (dataset-ref ds i)) list))))

(define/contract-out (indices->list indices) ;; noqa
  (-> indices/c (non-empty-listof exact-nonnegative-integer?))
  (if (tensor? indices) (tensor->list indices) indices))

(define items/c
  (flat-named-contract
   'stackable-tensor-items
   (lambda (v)
     (and (list? v)
          (pair? v)
          (for/and ([item (in-list v)])
            (and (list? item) (pair? item) (andmap tensor? item)))
          (let ([lead (first v)])
            (for/and ([item (in-list v)])
              (and (= (length item) (length lead))
                   (andmap (lambda (t u)
                             (and (equal? (shape t) (shape u))
                                  (equal? (tensor-device t) (tensor-device u))))
                           item
                           lead))))))))

(define/contract-out (default-collate items) ;; noqa
  (-> items/c any)
  (apply values
         (for/list ([field (in-range (length (first items)))])
           (stack (for/list ([item (in-list items)]) (list-ref item field))))))

;; chaperone-of?, not eq?: every importer holds its own contract chaperone
;; of the binding above, and a #:batch body sees one more on top
(define/contract-out (default-collate? v) ;; noqa
  (-> any/c boolean?)
  (chaperone-of? v default-collate))

(begin-for-syntax
  (define-splicing-syntax-class init-clause ;; noqa
    #:description "#:init clause"
    (pattern (~seq #:init f:init-formals body:expr ...)
      #:with formals #'f.formals
      #:with (id ...) #'(f.id ...)))

  (define-splicing-syntax-class length-clause ;; noqa
    #:description "#:length clause"
    (pattern (~seq #:length body:expr)))

  (define-splicing-syntax-class ref-clause ;; noqa
    #:description "#:ref clause"
    (pattern (~seq #:ref (index:id) body:expr ...+)))

  (define-splicing-syntax-class batch-clause ;; noqa
    #:description "#:batch clause"
    (pattern (~seq #:batch (indices:id collate:id) body:expr ...+)))

  (define-splicing-syntax-class device-clause ;; noqa
    #:description "#:device clause"
    (pattern (~seq #:device body:expr)))

  (define (non-bare-field fields bare?s) ;; noqa
    (for/first ([f (in-list fields)] [bare? (in-list bare?s)] #:unless bare?)
      f)))

(define-syntax (define-dataset stx)
  (syntax-parse stx
    [(_ name:id (field:ctor-formal ...)
        (~alt (~optional init:init-clause)
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))
              (~once len:length-clause #:name "#:length clause")
              (~once ref:ref-clause #:name "#:ref clause")
              (~optional batch:batch-clause)
              (~optional device:device-clause))
        ...)
     #:do [(define fields (syntax->list #'(field.id ...)))
           (define init? (attribute init))]
     #:fail-when (and init? (non-bare-field fields (attribute field.bare?)))
     "with #:init, a field is a bare identifier; defaults and keywords belong to the #:init formals"
     #:fail-when (and (attribute device) (not (attribute batch)) #'device.body)
     "#:device needs #:batch: a loader hands a device-resident index tensor to #:batch, and the default batch reads indices on the host"
     #:with sid (generate-temporary #'name)
     #:with sid? (format-id #'sid "~a?" #'sid)
     #:with name? (format-id #'name "~a?" #'name)
     #:with (field-acc ...) (for/list ([f (in-list fields)])
                              (format-id #'sid "~a-~a" #'sid f))
     #:with formals (if init? #'init.formals #'((~@ field.decl ...) ...))
     #:with (init-body ...) (if init? #'(init.body ...) #'())
     #:with (absent ...) (if init?
                             (filter (lambda (f)
                                       (not (member f (syntax->list #'(init.id ...))
                                                    bound-identifier=?)))
                                     fields)
                             '())
     #:with (batch-method ...)
     (if (attribute batch)
         #'((define (dataset-batch self batch.indices batch.collate)
              (let ([field.id (field-acc self)] ...)
                ((lambda (batch.indices batch.collate) batch.body ...)
                 batch.indices batch.collate))))
         #'())
     #:with (device-method ...)
     (if (attribute device)
         #'((define (dataset-device self)
              (let ([field.id (field-acc self)] ...)
                device.body)))
         #'())
     #:with export (contract-export stx #'name #'name? (attribute ctc) (attribute pred))
     #'(begin
         (struct sid (field.id ...)
           #:reflection-name 'name
           #:methods gen:dataset
           [(define (dataset-length self)
              (let ([field.id (field-acc self)] ...)
                len.body))
            (define (dataset-ref self ref.index)
              (let ([field.id (field-acc self)] ...)
                ((lambda (ref.index) ref.body ...) ref.index)))
            batch-method ...
            device-method ...])
         (define name? sid?)
         (define (name . formals)
           (let ([absent #f] ...)
             init-body ...
             (sid field.id ...)))
         export)]))
