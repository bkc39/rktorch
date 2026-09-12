#lang racket/base

(require (only-in racket/contract/base
                  -> ->i and/c any/c contract-out flat-named-contract listof
                  or/c unsupplied-arg?)
         (only-in racket/list first)
         (only-in "../foreign.rkt"
                  draw-seed generator? index-select make-generator narrow
                  randperm select tensor tensor-device tensor-shape tensor?
                  to)
         (only-in "../private/contract.rkt" define/contract-out)
         "dataset.rkt")

(provide (all-from-out "dataset.rkt"))

;; a contiguous ascending run of indices is a narrow, not a gather
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
                       (lambda (v) (and (tensor? v) (pair? (tensor-shape v))))))

(define (same-leading-dimension/c t)
  (flat-named-contract
   'same-leading-dimension
   (lambda (u) (and (batched/c u) (= (car (tensor-shape u)) (car (tensor-shape t)))))))

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
  #:length (car (tensor-shape (first tensors)))
  #:ref (i)
  (apply values (for/list ([t (in-list tensors)]) (select t 0 i)))
  #:device (tensor-device (first tensors))
  ;; the whole-batch path is default-collate's result computed natively;
  ;; a custom collate must see the items, as DataLoader's collate_fn does
  #:batch (indices collate)
  (cond
    [(not (default-collate? collate))
     (collate (for/list ([i (in-list (indices->list indices))])
                (for/list ([t (in-list tensors)]) (select t 0 i))))]
    [else
     (define n (car (tensor-shape (first tensors))))
     (define start (run-start indices n))
     (apply values
            (cond
              [start
               (for/list ([t (in-list tensors)])
                 (narrow t 0 start (length indices)))]
              [else
               (define index
                 (if (tensor? indices) indices (tensor indices #:dtype 'int64)))
               (for/list ([t (in-list tensors)])
                 (index-select t 0 (to index (tensor-device t))))]))]))

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
       "a shuffled loader needs a non-empty dataset, as RandomSampler does"
       (or (unsupplied-arg? shuffle?)
           (not shuffle?)
           (positive? (dataset-length ds)))
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
  (if g (draw-seed #:generator g) (draw-seed))
  (define sampler
    (and (dataloader-shuffle? loader) (or g (make-generator (draw-seed)))))
  (define perm #f)
  (define (permutation!)
    (unless perm
      (define drawn (randperm n #:generator sampler))
      (define dev (dataset-device ds))
      (set! perm (if dev (to drawn dev) drawn)))
    perm)
  (define count (batch-count loader))
  (define remainder-drawn? #f)
  (define (finish!)
    (when (and sampler (not remainder-drawn?))
      (permutation!)
      (set! remainder-drawn? #t)
      (void (randperm n #:generator sampler))))
  (define partial-last?
    (and (not (dataloader-drop-last? loader)) (positive? (remainder n b))))
  (define (batch k)
    (define start (* k b))
    (define len (min b (- n start)))
    (define indices
      (if sampler
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
