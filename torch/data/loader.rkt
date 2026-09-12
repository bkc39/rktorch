#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i any any/c contract-out flat-named-contract listof
                  non-empty-listof or/c)
         (only-in racket/generic define-generics)
         (only-in racket/list first)
         (only-in "../foreign.rkt"
                  device? draw-seed generator? index-select make-generator
                  narrow randperm select stack tensor tensor-device
                  tensor-dtype tensor-shape tensor->list tensor? to)
         (only-in "../private/contract.rkt" define/contract-out))

(provide gen:dataset
         (contract-out
          [dataset? (-> any/c boolean?)]
          [dataset-length (-> dataset? exact-nonnegative-integer?)]
          [dataset-ref (-> dataset? exact-nonnegative-integer? any)]
          [dataset-batch (-> dataset? indices/c collate/c any)]
          [dataset-device (-> dataset? (or/c device? #f))]))

(define index-tensor/c
  (flat-named-contract
   'int64-vector
   (lambda (v)
     (and (tensor? v) (eq? (tensor-dtype v) 'int64) (= 1 (length (tensor-shape v)))))))
(define indices/c (or/c (listof exact-nonnegative-integer?) index-tensor/c))
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
  (collate (for/list ([i (in-list (index-list indices))])
             (call-with-values (lambda () (dataset-ref ds i)) list))))

(define (index-list indices)
  (if (tensor? indices) (tensor->list indices) indices))

;; a contiguous ascending run of indices is a narrow, not a gather
(define (run-start indices n)
  (cond
    [(tensor? indices) #f]
    [(null? indices) #f]
    [else
     (define start (first indices))
     (and (for/and ([i (in-list indices)] [k (in-naturals start)]) (= i k))
          (<= (+ start (length indices)) n)
          start)]))

;; the struct's name goes to the variadic constructor below; the plain
;; constructor stays internal
(struct tensor-dataset (tensors) ;; noqa
  #:constructor-name make-tensor-dataset
  #:omit-define-syntaxes
  #:methods gen:dataset
  [(define (dataset-length self)
     (car (tensor-shape (first (tensor-dataset-tensors self)))))
   (define (dataset-ref self i)
     (apply values
            (for/list ([t (in-list (tensor-dataset-tensors self))])
              (select t 0 i))))
   (define (dataset-device self)
     (tensor-device (first (tensor-dataset-tensors self))))
   ;; the whole-batch path is default-collate's result computed natively;
   ;; a custom collate must see the items, as DataLoader's collate_fn does.
   ;; chaperone-of?, not eq?: the exported default-collate is the contract's
   ;; chaperone of the one bound here
   (define (dataset-batch self indices collate)
     (cond
       [(not (chaperone-of? collate default-collate))
        (batch-by-ref self indices collate)]
       [else
        (define ts (tensor-dataset-tensors self))
        (define start (run-start indices (dataset-length self)))
        (apply values
               (cond
                 [start
                  (for/list ([t (in-list ts)])
                    (narrow t 0 start (length indices)))]
                 [else
                  (define index
                    (if (tensor? indices)
                        indices
                        (tensor indices #:dtype 'int64)))
                  (for/list ([t (in-list ts)])
                    (index-select t 0 (to index (tensor-device t))))]))]))])

(define batched/c
  (flat-named-contract 'batched-tensor
                       (lambda (v) (and (tensor? v) (pair? (tensor-shape v))))))

(define (same-leading-dimension/c t)
  (flat-named-contract
   'same-leading-dimension
   (lambda (u) (and (batched/c u) (= (car (tensor-shape u)) (car (tensor-shape t)))))))

(define/contract-out (tensor-dataset t . more) ;; noqa
  (->i ([t batched/c])
       #:rest [more (t) (listof (same-leading-dimension/c t))]
       [result dataset?])
  (make-tensor-dataset (cons t more)))

(provide (contract-out [tensor-dataset? (-> any/c boolean?)]))

(define/contract-out (default-collate items) ;; noqa
  (-> (non-empty-listof (non-empty-listof tensor?)) any)
  (apply values
         (for/list ([field (in-range (length (first items)))])
           (stack (for/list ([item (in-list items)]) (list-ref item field))))))

(struct dataloader (dataset batch-size shuffle? drop-last? collate generator) ;; noqa
  #:constructor-name make-dataloader
  #:omit-define-syntaxes)

(define/contract-out (dataloader ds ;; noqa
                                 #:batch-size [batch-size 1]
                                 #:shuffle? [shuffle? #f]
                                 #:drop-last? [drop-last? #f]
                                 #:collate [collate default-collate]
                                 #:generator [generator #f])
  (->* [dataset?]
       [#:batch-size exact-positive-integer?
        #:shuffle? boolean?
        #:drop-last? boolean?
        #:collate collate/c
        #:generator (or/c generator? #f)]
       dataloader?)
  (make-dataloader ds batch-size shuffle? drop-last? collate generator))

(provide (contract-out [dataloader? (-> any/c boolean?)]))

(define/contract-out (dataloader-length loader) ;; noqa
  (-> dataloader? exact-nonnegative-integer?)
  (batch-count loader))

(define (batch-count loader)
  (define n (dataset-length (dataloader-dataset loader)))
  (define b (dataloader-batch-size loader))
  (if (dataloader-drop-last? loader)
      (quotient n b)
      (quotient (+ n b -1) b)))

;; One traversal draws what one DataLoader iterator draws, in its order:
;; the base seed on creation, the permutation when the first batch is
;; asked for, and the remainder permutation RandomSampler discards once
;; the first is used up: before a final partial batch, else on exhaustion.
(define (epoch-batches loader)
  (define ds (dataloader-dataset loader))
  (define n (dataset-length ds))
  (define b (dataloader-batch-size loader))
  (define g (dataloader-generator loader))
  (if g (draw-seed #:generator g) (draw-seed))
  (define sampler
    (and (dataloader-shuffle? loader) (or g (make-generator (draw-seed)))))
  (define perm
    (and sampler
         (let ([drawn (randperm n #:generator sampler)]
               [dev (dataset-device ds)])
           (if dev (to drawn dev) drawn))))
  (define count (batch-count loader))
  (define remainder-drawn? #f)
  (define (finish!)
    (when (and sampler (not remainder-drawn?))
      (set! remainder-drawn? #t)
      (void (randperm n #:generator sampler))))
  (define partial-last?
    (and (not (dataloader-drop-last? loader)) (positive? (remainder n b))))
  (define (batch k)
    (when (and partial-last? (= k (sub1 count)))
      (finish!))
    (define start (* k b))
    (define len (min b (- n start)))
    (define indices
      (if perm
          (narrow perm 0 start len)
          (for/list ([i (in-range start (+ start len))]) i)))
    (dataset-batch ds indices (dataloader-collate loader)))
  (values count batch finish!))

(define/contract-out (in-dataloader loader) ;; noqa
  (-> dataloader? sequence?)
  (make-do-sequence
   (lambda ()
     (define-values (count batch finish!) (epoch-batches loader))
     (values batch
             add1
             0
             (lambda (k)
               (or (< k count)
                   (begin
                     (finish!)
                     #f)))
             #f
             #f))))

(define/contract-out (in-epochs loader n-epochs) ;; noqa
  (-> dataloader? exact-nonnegative-integer? sequence?)
  (make-do-sequence
   (lambda ()
     (define batch #f)
     (define count 0)
     (define finish! void)
     ;; every epoch starts, and so draws, even one with no batches
     (define (start-from e)
       (cond
         [(>= e n-epochs) e]
         [else
          (define-values (c b f) (epoch-batches loader))
          (set! count c)
          (set! batch b)
          (set! finish! f)
          (cond
            [(zero? c)
             (finish!)
             (start-from (add1 e))]
            [else e])]))
     (values (lambda (pos)
               (call-with-values (lambda () (batch (cdr pos)))
                                 (lambda vals (apply values (car pos) vals))))
             (lambda (pos)
               (define k (add1 (cdr pos)))
               (cond
                 [(< k count) (cons (car pos) k)]
                 [else
                  (finish!)
                  (cons (start-from (add1 (car pos))) 0)]))
             (cons (start-from 0) 0)
             (lambda (pos) (< (car pos) n-epochs))
             #f
             #f))))
