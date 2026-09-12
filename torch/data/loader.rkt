#lang racket/base

(require (only-in racket/contract/base
                  -> ->* any any/c contract-out non-empty-listof)
         (only-in racket/generic define-generics)
         (only-in racket/list first)
         (only-in "../foreign.rkt"
                  draw-seed generator? index-select make-generator narrow
                  randperm select stack tensor tensor-device tensor-shape
                  tensor->list tensor? to)
         (only-in "../private/contract.rkt" define/contract-out))

(provide gen:dataset
         dataset-length
         dataset-ref
         dataset-batch
         dataset-device
         (contract-out [dataset? (-> any/c boolean?)]))

;; A map-style dataset: an item per index, and a batch for a run of indices.
;; The batch method is what a loader calls; the fallback collates the items
;; one by one, and tensor-backed datasets override it with one native op.
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
(struct tensor-dataset (tensors)
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

(define/contract-out (tensor-dataset t . more) ;; noqa
  (-> tensor? tensor? ... dataset?)
  (define n (car (tensor-shape t)))
  (for ([u (in-list more)])
    (unless (= (car (tensor-shape u)) n)
      (raise-arguments-error 'tensor-dataset
                             "every tensor must share the first dimension"
                             "expected" n
                             "given" (car (tensor-shape u)))))
  (make-tensor-dataset (cons t more)))

(provide (contract-out [tensor-dataset? (-> any/c boolean?)]))

;; torch's default_collate for tensor fields: one stack per field
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
        #:collate (-> (non-empty-listof list?) any)
        #:generator generator?]
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

;; What one DataLoader epoch draws, so a seeded loader replays its batch
;; order: with a generator, an int64 (the iterator's base seed), the
;; permutation, and the trailing permutation RandomSampler draws for its
;; remainder and discards; without one, two int64s from the global stream,
;; the second seeding a fresh generator that draws the two permutations.
(define (epoch-permutation loader n)
  (define g (dataloader-generator loader))
  (define gen
    (cond
      [g (draw-seed #:generator g) g]
      [else (draw-seed) (make-generator (draw-seed))]))
  (begin0 (randperm n #:generator gen)
    (randperm n #:generator gen)))

;; One traversal is one epoch: a permutation is drawn when the traversal
;; starts and each position yields the batch's fields as values.
(define (epoch-batches loader)
  (define ds (dataloader-dataset loader))
  (define n (dataset-length ds))
  (define b (dataloader-batch-size loader))
  ;; the permutation moves to the dataset's device once: a per-batch index
  ;; copy would make the host wait for the stream on every step
  (define perm
    (cond
      [(dataloader-shuffle? loader)
       (define drawn (epoch-permutation loader n))
       (define dev (dataset-device ds))
       (if dev (to drawn dev) drawn)]
      [else #f]))
  (define count (batch-count loader))
  (define (batch k)
    (define start (* k b))
    (define len (min b (- n start)))
    (define indices
      (if perm
          (narrow perm 0 start len)
          (for/list ([i (in-range start (+ start len))]) i)))
    (dataset-batch ds indices (dataloader-collate loader)))
  (values count batch))

(define/contract-out (in-dataloader loader) ;; noqa
  (-> dataloader? sequence?)
  (make-do-sequence
   (lambda ()
     (define-values (count batch) (epoch-batches loader))
     (values batch add1 0 (lambda (k) (< k count)) #f #f))))

;; (for ([(epoch xb yb) (in-epochs loader n)]) ...): epochs by number, each
;; a fresh traversal, so the generator's stream continues across them
(define/contract-out (in-epochs loader n-epochs) ;; noqa
  (-> dataloader? exact-nonnegative-integer? sequence?)
  (make-do-sequence
   (lambda ()
     (define batch #f)
     (define count 0)
     (define (start-epoch!)
       (define-values (c b) (epoch-batches loader))
       (set! count c)
       (set! batch b))
     (when (> n-epochs 0) (start-epoch!))
     (values (lambda (pos)
               (call-with-values (lambda () (batch (cdr pos)))
                                 (lambda vals (apply values (car pos) vals))))
             (lambda (pos)
               (define k (add1 (cdr pos)))
               (cond
                 [(< k count) (cons (car pos) k)]
                 [else
                  (define e (add1 (car pos)))
                  (when (< e n-epochs) (start-epoch!))
                  (cons e 0)]))
             (cons 0 0)
             (lambda (pos) (and (< (car pos) n-epochs) (< (cdr pos) count)))
             #f
             #f))))
