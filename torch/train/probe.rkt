#lang racket/base

(require (only-in racket/bool xor)
         (only-in racket/contract/base
                  -> ->* ->i >/c >=/c any/c cons/c contract-out flat-named-contract listof
                  non-empty-listof or/c procedure-arity-includes/c real-in unsupplied-arg?)
         (only-in racket/generic define/generic)
         (prefix-in list: (only-in racket/list argmax))
         (only-in racket/match match match* match-define)
         (only-in threading ~>)
         (only-in "../data/dataset.rkt"
                  dataset-batch dataset-device dataset-length dataset-ref dataset? gen:dataset)
         (only-in "../data/loader.rkt" dataloader in-dataloader tensor-dataset)
         (only-in "../foreign.rkt"
                  * + - / @ arange argmax backward! cat clamp default-device detach device device/c
                  dtype eq exp gen:sized index-fill! index-select item length make-generator max min
                  narrow nonzero numel ones randperm requires-grad! reshape scatter! scatter-add!
                  seed/c select shape sort sqrt stack sum take tensor->list tensor? to to-dtype topk
                  transpose with-no-grad zero-grad! zeros)
         (prefix-in g: (only-in "../generated.rkt" linalg-vector-norm mean-dim sum-dim-intlist))
         (only-in "../nn/layer.rkt" layer? parameters)
         (only-in "../nn/loss.rkt" cross-entropy)
         (only-in "../nn/metrics.rkt" accuracy)
         (only-in "../nn/mode.rkt" call-with-eval-mode)
         (only-in "../nn/private/tensor-checks.rkt"
                  float-dtype? in-range? labels/c rank-of rows same-device? same-rows?)
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "private/copy.rkt" copy-of)
         (only-in "private/lbfgs.rkt" lbfgs-minimize!)
         (only-in "private/types.rkt" define-named-contract))

(define-named-contract probe-pool/c (or/c 'cls 'mean 'cls+mean #f))

(define (integer-dtype? dt)
  (and (memq dt '(uint8 int8 int16 int32 int64)) #t))

(define probe-features/c
  (flat-named-contract 'probe-features
                       (lambda (v)
                         (and (tensor? v)
                              (= 2 (rank-of v))
                              (memq (dtype v) '(float32 float64))
                              (positive? (rows v))))))

(define weight-decays/c (non-empty-listof (>=/c 0)))

(struct probe-dataset (features labels classes indices inner)
  #:constructor-name make-probe-dataset
  #:omit-define-syntaxes
  #:methods gen:dataset
  [(define/generic inner-length dataset-length)
   (define/generic inner-ref dataset-ref)
   (define/generic inner-batch dataset-batch)
   (define/generic inner-device dataset-device)
   (define (dataset-length ds) (inner-length (probe-dataset-inner ds)))
   (define (dataset-ref ds i) (inner-ref (probe-dataset-inner ds) i))
   (define (dataset-batch ds indices collate) (inner-batch (probe-dataset-inner ds) indices collate))
   (define (dataset-device ds) (inner-device (probe-dataset-inner ds)))]
  #:methods gen:sized
  [(define (length ds) (dataset-length (probe-dataset-inner ds)))])

(provide (contract-out
          [probe-dataset? (-> any/c boolean?)]
          [probe-dataset-features (-> probe-dataset? tensor?)]
          [probe-dataset-labels (-> probe-dataset? tensor?)]
          [probe-dataset-classes (-> probe-dataset? exact-positive-integer?)]
          [probe-dataset-indices (-> probe-dataset? (or/c #f tensor?))]))

(define (width v)
  (match-define (list _ d) (shape v))
  d)

(define (same-features? fit other)
  (define (features-of ds) (probe-dataset-features ds))
  (or (not other)
      (let ([f (features-of fit)] [g (features-of other)])
        (and (= (width f) (width g)) (eq? (dtype f) (dtype g)) (same-device? f g)))))

(define (k-fits? fit query k)
  (define n (length fit))
  (if query (<= k n) (< k n)))

(define (classes-fit? fit query)
  (or (not query) (<= (probe-dataset-classes query) (probe-dataset-classes fit))))

(define (cached features labels classes indices)
  (define kept (detach features))
  (make-probe-dataset kept labels classes indices (tensor-dataset kept labels)))

(define (encoder-device encoder dataset)
  (match (and (layer? encoder) (parameters encoder))
    [(cons p _) (device p)]
    [_ (or (dataset-device dataset) (default-device))]))

(define (tokens-mean tokens)
  (match-define (list _ t _) (shape tokens))
  (g:mean-dim (narrow tokens 1 1 (- t 1)) '(1) #f #f))

(define (several? t)
  (< 1 t))

(define (pooled out pool)
  (match* (pool (shape out))
    [(#f (list _ _)) out]
    [('cls (list _ _ _)) (copy-of (select out 1 0))]
    [('mean (list _ (? several?) _)) (tokens-mean out)]
    [('cls+mean (list _ (? several?) _)) (cat (list (select out 1 0) (tokens-mean out)) 1)]
    [(_ dims)
     (raise-arguments-error 'probe-dataset
                            (if pool
                                "the encoder's output is not [N, T, D] tokens with T > 1"
                                "the encoder's output is not [N, D] features")
                            "pool" pool
                            "shape" dims)]))

(define (batch-features out labels pool)
  (unless (and (tensor? out) (float-dtype? (dtype out)))
    (raise-arguments-error 'probe-dataset "the encoder's output is not a floating-point tensor"
                           "output" out))
  (unless (and (tensor? labels) (= 1 (rank-of labels)) (integer-dtype? (dtype labels)))
    (raise-arguments-error 'probe-dataset "a batch's labels are not an [N] integer tensor"
                           "labels" labels))
  (define features (pooled (to-dtype out 'float32) pool))
  (unless (= (rows features) (rows labels))
    (raise-arguments-error 'probe-dataset "the encoder's output has one row per label"
                           "rows" (rows features)
                           "labels" (rows labels)))
  features)

(define (encode-all dataset forward pool batch-size encoder-place cache-place)
  (with-no-grad
    (for/lists (_features _labels)
               ([batch (~> (dataloader dataset #:batch-size batch-size)
                           in-dataloader
                           in-values-sequence)])
      (match batch
        [(list* images ys _)
         (values (~> (to images encoder-place) forward (batch-features ys pool) (to cache-place))
                 (~> ys (to-dtype 'int64) (to cache-place)))]
        [_ (raise-arguments-error 'probe-dataset "a batch is not images and labels"
                                  "values" (length batch))]))))

(define/contract-out (probe-dataset encoder dataset ;; noqa
                                    #:pool [pool 'cls]
                                    #:forward [forward #f]
                                    #:batch-size [batch-size 1024]
                                    #:device [device #f]
                                    #:classes [classes #f])
  (->* [(or/c layer? procedure?) dataset?]
       [#:pool probe-pool/c
        #:forward (or/c #f (procedure-arity-includes/c 1))
        #:batch-size exact-positive-integer?
        #:device (or/c #f device/c)
        #:classes (or/c #f exact-positive-integer?)]
       probe-dataset?)
  (define encoder-place (encoder-device encoder dataset))
  (define (pass)
    (encode-all dataset (or forward encoder) pool batch-size encoder-place
                (or device encoder-place)))
  (define-values (features labels)
    (if (layer? encoder) (call-with-eval-mode encoder pass) (pass)))
  (when (null? features)
    (raise-arguments-error 'probe-dataset "the dataset has no items" "dataset" dataset))
  (define all-labels (cat labels))
  (define class-count (or classes (+ 1 (item (max all-labels)))))
  (unless (in-range? all-labels class-count)
    (match-define (list lo hi) (tensor->list (stack (list (min all-labels) (max all-labels)))))
    (raise-arguments-error 'probe-dataset "a label is outside [0, classes)"
                           "classes" class-count
                           "smallest" lo
                           "largest" hi))
  (cached (cat features) all-labels class-count #f))

(define/contract-out (tensors->probe-dataset features labels #:classes [classes #f]) ;; noqa
  (->i ([features probe-features/c] [labels labels/c])
       (#:classes [classes (or/c #f exact-positive-integer?)])
       #:pre/name (features labels) "one label per feature row"
       (same-rows? features labels)
       #:pre/name (features labels) "features and labels on one device"
       (same-device? features labels)
       #:pre/name (labels classes) "labels in [0, classes)"
       (in-range? labels (or (supplied-or classes #f) +inf.0))
       [result probe-dataset?])
  (cached features labels (or classes (+ 1 (item (max labels)))) #f))

(define (subset ds indices)
  (define index (to indices (device (probe-dataset-features ds))))
  (cached (index-select (probe-dataset-features ds) 0 index)
          (index-select (probe-dataset-labels ds) 0 index)
          (probe-dataset-classes ds)
          index))

(define (class-draw labels c per-class generator)
  (define members (~> (eq labels c) nonzero (reshape -1)))
  (define n (numel members))
  (when (< n per-class)
    (raise-arguments-error 'probe-split "a class has fewer rows than select-per-class"
                           "class" c
                           "rows" n
                           "select-per-class" per-class))
  (index-select members 0 (narrow (randperm n #:generator generator) 0 0 per-class)))

(define/contract-out (probe-split ds #:select-per-class [per-class 500] #:seed [seed 0]) ;; noqa
  (->* [probe-dataset?] [#:select-per-class exact-positive-integer? #:seed seed/c]
       (values probe-dataset? probe-dataset?))
  (define labels (to (probe-dataset-labels ds) 'cpu))
  (define generator (make-generator seed))
  (define-values (chosen _order)
    (~> (for/list ([c (in-range (probe-dataset-classes ds))])
          (class-draw labels c per-class generator))
        cat
        sort))
  (define kept (ones (numel labels) #:dtype 'int64 #:device (device labels)))
  (index-fill! kept 0 chosen 0)
  (values (subset ds (reshape (nonzero kept) -1)) (subset ds chosen)))

(struct linear-probe (weight bias weight-decay sweep mean std classes) ;; noqa
  #:constructor-name make-linear-probe
  #:omit-define-syntaxes)

(provide (contract-out
          [linear-probe? (-> any/c boolean?)]
          [linear-probe-weight (-> linear-probe? tensor?)]
          [linear-probe-bias (-> linear-probe? tensor?)]
          [linear-probe-weight-decay (-> linear-probe? (>=/c 0))]
          [linear-probe-sweep
           (-> linear-probe? (or/c #f (listof (cons/c (>=/c 0) (real-in 0 1)))))]
          [linear-probe-mean (-> linear-probe? tensor?)]
          [linear-probe-std (-> linear-probe? tensor?)]
          [linear-probe-classes (-> linear-probe? exact-positive-integer?)]))

(define (probe-fits? probe ds)
  (define w (linear-probe-weight probe))
  (define f (probe-dataset-features ds))
  (and (= (rows w) (width f)) (eq? (dtype w) (dtype f)) (same-device? w f)))

(define (standardiser x)
  ;; torch.std accumulates a float32 column in double on the CPU; MPS has no float64
  (define host (~> x (to 'cpu) (to-dtype 'float64)))
  (define centred (- host (g:mean-dim host '(0) #f #f)))
  (values (g:mean-dim x '(0) #f #f)
          (~> (* centred centred)
              (g:sum-dim-intlist '(0) #f #f)
              (/ (- (rows x) 1))
              sqrt
              (to-dtype (dtype x))
              (to (device x))
              (clamp #:min 1e-6))))

(define (standardised x mean std)
  (/ (- x mean) std))

(define (probe-objective x y weight bias weight-decay)
  (+ (~> (@ x weight) (+ bias) (cross-entropy y))
     (* (* 0.5 weight-decay) (sum (* weight weight)))))

(define (fit-weights x y classes weight-decay max-iter)
  (define (parameter . dims)
    (requires-grad! (apply zeros dims #:dtype (dtype x) #:device (device x))))
  (define weight (parameter (width x) classes))
  (define bias (parameter classes))
  (define (closure)
    (zero-grad! weight)
    (zero-grad! bias)
    (define loss (probe-objective x y weight bias weight-decay))
    (backward! loss)
    loss)
  (lbfgs-minimize! (list weight bias) closure
                   #:lr 1.0
                   #:max-iter max-iter
                   #:tolerance-grad 1e-7
                   #:tolerance-change 1e-10
                   #:history-size 20
                   #:line-search 'strong-wolfe)
  (values (detach weight) (detach bias)))

(struct swept (weight-decay weight bias score))

(define (scored weight bias x labels)
  (~> (@ x weight) (+ bias) (accuracy labels #:validate-args? #f) item))

(define/contract-out (linear-probe fit [select #f] ;; noqa
                                   #:weight-decays
                                   [weight-decays '(1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1)]
                                   #:weight-decay [weight-decay #f]
                                   #:max-iter [max-iter 1000])
  (->i ([fit probe-dataset?])
       ([select (or/c #f probe-dataset?)]
        #:weight-decays [weight-decays weight-decays/c]
        #:weight-decay [weight-decay (or/c #f (>=/c 0))]
        #:max-iter [max-iter exact-positive-integer?])
       #:pre/name (select weight-decay) "a select split to sweep on, or one #:weight-decay, not both"
       (xor (supplied-or select #f) (supplied-or weight-decay #f))
       #:pre/name (select weight-decays) "#:weight-decays needs a select split"
       (or (supplied-or select #f) (unsupplied-arg? weight-decays))
       #:pre/name (fit select) "fit and select of one feature width, dtype and device"
       (same-features? fit (supplied-or select #f))
       #:pre/name (fit select) "select classes within the fit's"
       (classes-fit? fit (supplied-or select #f))
       #:pre/name (fit) "at least two fit rows"
       (<= 2 (length fit))
       [result linear-probe?])
  (define x (probe-dataset-features fit))
  (define-values (mean std) (standardiser x))
  (define z (standardised x mean std))
  (define classes (probe-dataset-classes fit))
  (define (fitted wd)
    (fit-weights z (probe-dataset-labels fit) classes wd max-iter))
  (cond
    [select
     (define z-select (standardised (probe-dataset-features select) mean std))
     (define fits
       (for/list ([wd (in-list weight-decays)])
         (define-values (w b) (fitted wd))
         (swept wd w b (scored w b z-select (probe-dataset-labels select)))))
     (match-define (swept wd w b _) (list:argmax swept-score fits))
     (make-linear-probe w b wd
                        (for/list ([fit (in-list fits)])
                          (cons (swept-weight-decay fit) (swept-score fit)))
                        mean std classes)]
    [else
     (define-values (w b) (fitted weight-decay))
     (make-linear-probe w b weight-decay #f mean std classes)]))

(define/contract-out (probe-accuracy probe ds) ;; noqa
  (->i ([probe linear-probe?] [ds probe-dataset?])
       #:pre/name (probe ds) "features of the probe's width, dtype and device"
       (probe-fits? probe ds)
       #:pre/name (probe ds) "classes within the probe's"
       (<= (probe-dataset-classes ds) (linear-probe-classes probe))
       [result (real-in 0 1)])
  (scored (linear-probe-weight probe)
          (linear-probe-bias probe)
          (standardised (probe-dataset-features ds) (linear-probe-mean probe) (linear-probe-std probe))
          (probe-dataset-labels ds)))

(define (normalised x)
  (/ x (~> (g:linalg-vector-norm x 2.0 '(1) #t #f) (clamp #:min 1e-12))))

(define (votes fit-x fit-labels classes query-x leave-one-out? k temperature batch-size)
  (define fit-t (transpose fit-x 0 1))
  (define n (rows query-x))
  (for/list ([start (in-range 0 n batch-size)])
    (define m (min batch-size (- n start)))
    (define sim (@ (narrow query-x 0 start m) fit-t))
    (when leave-one-out?
      (scatter! sim 1
                (~> (arange start (+ start m) #:dtype 'int64 #:device (device sim)) (reshape m 1))
                -inf.0))
    (define-values (top idx) (topk sim k #:dim 1))
    (define tally (zeros m classes #:dtype (dtype sim) #:device (device sim)))
    (scatter-add! tally 1 (take fit-labels idx)
                  (~> (- top (narrow top 1 0 1)) (/ temperature) exp))
    tally))

(define/contract-out (knn-accuracy fit [query #f] ;; noqa
                                   #:k [k 20]
                                   #:temperature [temperature 0.07]
                                   #:standardize? [standardize? #t]
                                   #:batch-size [batch-size 1000])
  (->i ([fit probe-dataset?])
       ([query (or/c #f probe-dataset?)]
        #:k [k exact-positive-integer?]
        #:temperature [temperature (>/c 0)]
        #:standardize? [standardize? boolean?]
        #:batch-size [batch-size exact-positive-integer?])
       #:pre/name (fit query) "query of the fit's feature width, dtype and device"
       (same-features? fit (supplied-or query #f))
       #:pre/name (fit query k) "k at most the fit rows, less one when leaving one out"
       (k-fits? fit (supplied-or query #f) (supplied-or k 20))
       #:pre/name (fit query) "query classes within the fit's"
       (classes-fit? fit (supplied-or query #f))
       #:pre/name (fit standardize?) "at least two fit rows when standardizing"
       (or (not (supplied-or standardize? #t)) (<= 2 (length fit)))
       [result (real-in 0 1)])
  (~> (knn-predictions fit query #:k k #:temperature temperature #:standardize? standardize?
                       #:batch-size batch-size)
      (accuracy (probe-dataset-labels (or query fit)) #:validate-args? #f)
      item))

(define (knn-predictions fit query
                         #:k k
                         #:temperature temperature
                         #:standardize? standardize?
                         #:batch-size [batch-size 1000])
  (define x (probe-dataset-features fit))
  (define prepared
    (cond
      [standardize?
       (define-values (mean std) (standardiser x))
       (lambda (v) (normalised (standardised v mean std)))]
      [else normalised]))
  (define fit-x (prepared x))
  (define query-x (if query (prepared (probe-dataset-features query)) fit-x))
  (with-no-grad
    (~> (votes fit-x (probe-dataset-labels fit) (probe-dataset-classes fit) query-x (not query)
               k temperature batch-size)
        cat
        (argmax 1))))

(module+ private
  (provide knn-predictions pooled probe-objective))
