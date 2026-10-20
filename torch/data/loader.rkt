#lang racket/base

(require (only-in racket/contract/base
                  -> ->i and/c any/c contract-out flat-named-contract listof
                  or/c unsupplied-arg?)
         (only-in racket/list first)
         (only-in racket/match == match)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  device device-type draw-seed generator? index-select
                  make-generator narrow gen:sized length randperm select shape
                  tensor tensor? to)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "dataset.rkt"
                  collate/c dataset-batch dataset-device dataset-length
                  dataset-ref dataset? default-collate default-collate?
                  define-dataset gen:dataset indices->list indices/c)
         (only-in "workers.rkt"
                  built raised receive-batch request-batch! start-workers
                  stop-workers!))

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
   (lambda (u) (and (tensor? u) (equal? (device u) (device t))))))

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
  #:device (device (first tensors))
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
               (define dev (device (first tensors)))
               (define index
                 (if (tensor? indices)
                     (to indices dev)
                     (tensor indices #:dtype 'int64 #:device dev)))
               (for/list ([t (in-list tensors)])
                 (index-select t 0 index))]))]))

(struct dataloader (dataset batch-size shuffle? drop-last? collate generator ;; noqa
                    workers prefetch)
  #:constructor-name make-dataloader
  #:omit-define-syntaxes
  #:methods gen:sized
  [(define (length loader) (batch-count loader))]
  #:property prop:sequence (lambda (loader) (in-dataloader loader)))

(define (host-dataset? ds)
  (define dev (dataset-device ds))
  (or (not dev) (eq? (device-type dev) 'cpu)))

(define/contract-out (dataloader ds ;; noqa
                                 #:batch-size [batch-size 1]
                                 #:shuffle? [shuffle? #f]
                                 #:drop-last? [drop-last? #f]
                                 #:collate [collate default-collate]
                                 #:generator [generator #f]
                                 #:workers [workers 0]
                                 #:prefetch [prefetch 2])
  (->i ([ds dataset?])
       (#:batch-size [batch-size exact-positive-integer?]
        #:shuffle? [shuffle? boolean?]
        #:drop-last? [drop-last? boolean?]
        #:collate [collate collate/c]
        #:generator [generator (or/c generator? #f)]
        #:workers [workers exact-nonnegative-integer?]
        #:prefetch [prefetch exact-positive-integer?])
       #:pre/name (ds shuffle?)
       "a shuffled loader needs a non-empty dataset within randperm's size range"
       (or (unsupplied-arg? shuffle?)
           (not shuffle?)
           (< 0 (dataset-length ds) (expt 2 63)))
       #:pre/name (ds workers)
       "workers build CPU tensors: a dataset on an accelerator takes #:workers 0"
       (or (unsupplied-arg? workers) (zero? workers) (host-dataset? ds))
       #:pre/name (workers prefetch)
       "#:prefetch counts batches per worker and needs #:workers above 0"
       (or (unsupplied-arg? prefetch)
           (and (not (unsupplied-arg? workers)) (positive? workers)))
       [result dataloader?])
  (make-dataloader ds batch-size shuffle? drop-last? collate generator
                   workers prefetch))

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

(define (short-last-batch? loader)
  (and (not (dataloader-drop-last? loader))
       (~> loader
           dataloader-dataset
           dataset-length
           (remainder (dataloader-batch-size loader))
           positive?)))

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

(define (sampler-draws loader)
  (define ds (dataloader-dataset loader))
  (define n (dataset-length ds))
  (define b (dataloader-batch-size loader))
  (define shuffle? (dataloader-shuffle? loader))
  (define sampler #f)
  (define (sampler!)
    (unless sampler
      (set! sampler (or (dataloader-generator loader) (make-generator (draw-seed)))))
    sampler)
  (define perm #f)
  (define (permutation!)
    (unless perm
      (define drawn (randperm n #:generator (sampler!)))
      (define dev (dataset-device ds))
      (set! perm (if dev (to drawn dev) drawn)))
    perm)
  (define discarded? #f)
  (define (finish!)
    (when (and shuffle? (not discarded?))
      (permutation!)
      (randperm n #:generator (sampler!))
      (set! discarded? #t)))
  (define (indices k)
    (define start (* k b))
    (define len (min b (- n start)))
    (if shuffle?
        (narrow (permutation!) 0 start len)
        (for/list ([i (in-range start (+ start len))]) i)))
  (values indices finish!))

(define (worker-epoch-batches loader)
  (define workers (dataloader-workers loader))
  (define base (draw-seed #:generator (dataloader-generator loader)))
  (define breaks (current-break-parameterization))
  (parameterize-break #f
    (define pool (start-workers (dataloader-dataset loader) (dataloader-collate loader)
                                workers base))
    (with-handlers ([(lambda (_v) #t)
                     (lambda (v)
                       (stop-workers! pool)
                       (raise v))])
      (call-with-break-parameterization
       breaks
       (lambda () (pool-batches loader pool (* workers (dataloader-prefetch loader))))))))

(define (pool-batches loader pool ahead)
  (define-values (indices finish!) (sampler-draws loader))
  (define count (batch-count loader))
  (define exhausting (if (short-last-batch? loader) (sub1 count) count))
  (define (refill! r)
    (define batch-indices (and (< r count) (indices r)))
    (cond
      [(and batch-indices (= r exhausting))
       (request-batch! pool r batch-indices)
       (finish!)]
      [batch-indices (request-batch! pool r batch-indices)]
      [(= r exhausting) (finish!)]
      [else (void)]))
  (for ([r (in-range ahead)])
    (refill! r))
  (define failure #f)
  (define (receive-and-refill! k breakable?)
    (begin0 (receive-batch pool k breakable?)
      (refill! (+ k ahead))
      (when (= k (sub1 count))
        (stop-workers! pool))))
  ;; a break between batch's return and the sequence recording its values
  ;; asks for the same k again
  (define taken #f)
  (define (receive-once! k breakable?)
    (match taken
      [(cons (== k) result) result]
      [_ (set! taken #f)
         (define result (receive-and-refill! k breakable?))
         (set! taken (cons k result))
         result]))
  (define (batch k)
    (define breakable? (break-enabled))
    (parameterize-break #f
      (match (or failure (receive-once! k breakable?))
        [(built values-list) (apply values values-list)]
        [(and (raised v) failed)
         (set! failure failed)
         (stop-workers! pool)
         (raise v)])))
  (define (exhausted!)
    (set! taken #f)
    (finish!)
    (stop-workers! pool))
  (values count batch exhausted!))

(define (traversal loader)
  (if (zero? (dataloader-workers loader))
      (epoch-batches loader)
      (worker-epoch-batches loader)))

(module+ private
  (provide sampler-draws traversal))

(define/contract-out (in-dataloader loader) ;; noqa
  (-> dataloader? sequence?)
  (make-do-sequence
   (lambda ()
     (define-values (count batch finish!) (traversal loader))
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
          (define-values (c b f) (traversal loader))
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
