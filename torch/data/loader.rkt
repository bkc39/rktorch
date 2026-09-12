#lang racket/base

(require (only-in racket/contract/base
                  -> ->i and/c any/c contract-out flat-named-contract listof
                  or/c unsupplied-arg?)
         (only-in racket/list first)
         (only-in "../foreign.rkt"
                  draw-seed generator? index-select make-generator narrow
                  randperm select shape tensor tensor-device tensor? to)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "dataset.rkt"
                  collate/c dataset-batch dataset-device dataset-length
                  dataset-ref dataset? default-collate default-collate?
                  define-dataset gen:dataset indices->list indices/c))

(provide (all-from-out "dataset.rkt"))

(define (run-start indices n)
  (cond
    [(tensor? indices) #f]
    [else
     (define start (first indices))
     (and (for/and ([i (in-list indices)] [k (in-naturals start)]) (= i k))
          (<= (+ start (length indices)) n)
          start)]))

(define batched/c
  (flat-named-contract 'batched-tensor
                       (lambda (v) (and (tensor? v) (pair? (shape v))))))

(define (same-leading-dimension/c t)
  (flat-named-contract
   'same-leading-dimension
   (lambda (u) (and (batched/c u) (= (car (shape u)) (car (shape t)))))))

(define (same-device/c t)
  (flat-named-contract
   'same-device
   (lambda (u) (and (tensor? u) (equal? (tensor-device u) (tensor-device t))))))

(define-dataset tensor-dataset (tensors) ;; noqa
  #:contract (->i ([t batched/c])
                  #:rest [more (t) (listof (and/c (same-leading-dimension/c t)
                                                  (same-device/c t)))]
                  [result tensor-dataset?])
  #:init (t . more)
  (set! tensors (cons t more))
  #:length (car (shape (first tensors)))
  #:ref (i)
  (apply values (for/list ([t (in-list tensors)]) (select t 0 i)))
  #:device (tensor-device (first tensors))
  #:batch (indices collate)
  (cond
    [(not (default-collate? collate))
     (collate (for/list ([i (in-list (indices->list indices))])
                (for/list ([t (in-list tensors)]) (select t 0 i))))]
    [else
     (define n (car (shape (first tensors))))
     (define start (run-start indices n))
     (apply values
            (cond
              [start
               (for/list ([t (in-list tensors)])
                 (narrow t 0 start (length indices)))]
              [else
               (define dev (tensor-device (first tensors)))
               (define index
                 (if (tensor? indices)
                     (to indices dev)
                     (tensor indices #:dtype 'int64 #:device dev)))
               (for/list ([t (in-list tensors)])
                 (index-select t 0 index))]))]))

(struct dataloader (dataset batch-size shuffle? drop-last? collate generator) ;; noqa
  #:constructor-name make-dataloader
  #:omit-define-syntaxes)

(define/contract-out (dataloader ds ;; noqa
                                 #:batch-size [batch-size 1]
                                 #:shuffle? [shuffle? #f]
                                 #:drop-last? [drop-last? #f]
                                 #:collate [collate default-collate]
                                 #:generator [generator #f])
  (->i ([ds dataset?])
       (#:batch-size [batch-size exact-positive-integer?]
        #:shuffle? [shuffle? boolean?]
        #:drop-last? [drop-last? boolean?]
        #:collate [collate collate/c]
        #:generator [generator (or/c generator? #f)])
       #:pre/name (ds shuffle?)
       "a shuffled loader needs a non-empty dataset within randperm's size range"
       (or (unsupplied-arg? shuffle?)
           (not shuffle?)
           (< 0 (dataset-length ds) (expt 2 63)))
       [result dataloader?])
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
  (draw-seed #:generator g)
  (define shuffle? (dataloader-shuffle? loader))
  ;; RandomSampler seeds its own generator on the first next, not on iter
  (define sampler #f)
  (define (sampler!)
    (unless sampler
      (set! sampler (or g (make-generator (draw-seed)))))
    sampler)
  (define perm #f)
  (define (permutation!)
    (unless perm
      (define drawn (randperm n #:generator (sampler!)))
      (define dev (dataset-device ds))
      (set! perm (if dev (to drawn dev) drawn)))
    perm)
  (define count (batch-count loader))
  (define remainder-drawn? #f)
  (define (finish!)
    (when (and shuffle? (not remainder-drawn?))
      (permutation!)
      (set! remainder-drawn? #t)
      (void (randperm n #:generator (sampler!)))))
  (define partial-last?
    (and (not (dataloader-drop-last? loader)) (positive? (remainder n b))))
  (define (batch k)
    (define start (* k b))
    (define len (min b (- n start)))
    (define indices
      (if shuffle?
          (narrow (permutation!) 0 start len)
          (for/list ([i (in-range start (+ start len))]) i)))
    (when (and partial-last? (= k (sub1 count)))
      (finish!))
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
     (define e 0)
     (define k 0)
     (define count 0)
     (define batch #f)
     (define finish! void)
     (define started? #f)
     ;; an epoch starts when its first batch is asked for, and the one
     ;; before it is finished then, so a loop that stops on another sequence
     ;; leaves the stream where its last batch left it
     (define (more?)
       (cond
         [(>= e n-epochs) #f]
         [(not started?)
          (define-values (c b f) (epoch-batches loader))
          (set! count c)
          (set! batch b)
          (set! finish! f)
          (set! k 0)
          (set! started? #t)
          (more?)]
         [(< k count) #t]
         [else
          (finish!)
          (set! e (add1 e))
          (set! started? #f)
          (more?)]))
     (values (lambda (_i)
               (call-with-values (lambda () (batch k))
                                 (lambda vals (apply values e vals))))
             (lambda (i)
               (set! k (add1 k))
               (add1 i))
             0
             (lambda (_i) (more?))
             #f
             #f))))
