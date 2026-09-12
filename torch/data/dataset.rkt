#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id generate-temporary)
                     ;; whole-module on purpose: the expansion needs bindings
                     ;; only-in would strip
                     syntax/parse/pre
                     "../private/definer.rkt")
         (only-in racket/contract/base
                  -> any any/c contract-out flat-named-contract
                  non-empty-listof or/c)
         (only-in racket/generic define-generics)
         (only-in racket/list first)
         (only-in "../foreign.rkt"
                  device? stack tensor-dtype tensor-shape tensor->list tensor?)
         (only-in "../private/contract.rkt" define/contract-out))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:dataset
         define-dataset
         indices/c
         collate/c
         (contract-out
          [dataset? (-> any/c boolean?)]
          [dataset-length (-> dataset? exact-nonnegative-integer?)]
          [dataset-ref (-> dataset? exact-nonnegative-integer? any)]
          [dataset-batch (-> dataset? indices/c collate/c any)]
          [dataset-device (-> dataset? (or/c device? #f))]))

(define index-tensor/c
  (flat-named-contract
   'non-empty-int64-vector
   (lambda (v)
     (and (tensor? v)
          (eq? (tensor-dtype v) 'int64)
          (= 1 (length (tensor-shape v)))
          (positive? (car (tensor-shape v)))))))
(define indices/c
  (or/c (non-empty-listof exact-nonnegative-integer?) index-tensor/c))
(define collate/c (-> (non-empty-listof list?) any))

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
   'rectangular-tensor-items
   (lambda (v)
     (and (list? v)
          (pair? v)
          (for/and ([item (in-list v)])
            (and (list? item) (pair? item) (andmap tensor? item)))
          (let ([n (length (first v))])
            (for/and ([item (in-list v)]) (= (length item) n)))))))

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

(define-syntax (define-dataset stx)
  (syntax-parse stx
    [(_ name:id (field:ctor-formal ...)
        (~alt (~optional (~seq #:init init:init-formals init-body:expr ...))
              (~optional (~seq #:contract ctc:expr))
              (~optional (~seq #:predicate pred:id))
              (~once (~seq #:length length-body:expr)
                     #:name "#:length clause")
              (~once (~seq #:ref (index:id) ref-body:expr ...+)
                     #:name "#:ref clause")
              (~optional (~seq #:batch (indices:id collate:id)
                               batch-body:expr ...+))
              (~optional (~seq #:device device-body:expr)))
        ...)
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
     (with-syntax ([sid struct-id]
                   [sid? (format-id struct-id "~a?" struct-id)]
                   [name? (format-id #'name "~a?" #'name)]
                   [formals (if init?
                                #'init.formals
                                #'((~@ field.decl ...) ...))]
                   [(init-body ...) (if init? #'(init-body ...) #'())]
                   [(absent ...)
                    (filter (lambda (f)
                              (not (member f init-ids bound-identifier=?)))
                            (if init? field-ids '()))]
                   [(field-acc ...)
                    (for/list ([f (in-list field-ids)])
                      (format-id struct-id "~a-~a" struct-id f))])
       (with-syntax ([(batch-method ...)
                      (if (attribute batch-body)
                          #'((define (dataset-batch self indices collate)
                               (let ([field.id (field-acc self)] ...)
                                 ((lambda (indices collate) batch-body ...)
                                  indices collate))))
                          #'())]
                     [(device-method ...)
                      (if (attribute device-body)
                          #'((define (dataset-device self)
                               (let ([field.id (field-acc self)] ...)
                                 device-body)))
                          #'())]
                     [export (contract-export stx #'name #'name?
                                              (attribute ctc)
                                              (attribute pred))])
         #'(begin
             (struct sid (field.id ...)
               #:reflection-name 'name
               #:methods gen:dataset
               [(define (dataset-length self)
                  (let ([field.id (field-acc self)] ...)
                    length-body))
                (define (dataset-ref self index)
                  (let ([field.id (field-acc self)] ...)
                    ((lambda (index) ref-body ...) index)))
                batch-method ...
                device-method ...])
             (define name? sid?)
             (define (name . formals)
               (let ([absent #f] ...)
                 init-body ...
                 (sid field.id ...)))
             export)))]))
