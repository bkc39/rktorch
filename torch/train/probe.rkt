#lang racket/base

(require (only-in racket/bool xor)
         (only-in racket/contract/base
                  -> ->* ->i >/c >=/c any/c cons/c contract-out flat-named-contract listof
                  non-empty-listof or/c procedure-arity-includes/c real-in unsupplied-arg?)
         (only-in racket/generic define/generic)
         (only-in racket/match match-define)
         (only-in "../data/dataset.rkt"
                  dataset-batch dataset-device dataset-length dataset-ref dataset? gen:dataset)
         (only-in "../foreign.rkt"
                  device device/c dtype gen:sized length max min numel seed/c shape stack
                  tensor->list tensor?)
         (only-in "../nn/layer.rkt" layer?)
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "private/types.rkt" define-named-contract))

(define-named-contract probe-pool/c (or/c 'cls 'mean 'cls+mean #f))

(define (rank-of v)
  (length (shape v)))

(define probe-features/c
  (flat-named-contract 'probe-features
                       (lambda (v)
                         (and (tensor? v)
                              (= 2 (rank-of v))
                              (memq (dtype v) '(float32 float64))
                              (positive? (car (shape v)))))))

(define labels/c
  (flat-named-contract 'labels
                       (lambda (v)
                         (and (tensor? v) (= 1 (rank-of v)) (eq? 'int64 (dtype v))))))

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

(define (rows v)
  (car (shape v)))

(define (width v)
  (cadr (shape v)))

(define (same-rows? features labels)
  (= (rows features) (rows labels)))

(define (same-device? a b)
  (equal? (device a) (device b)))

;; one host read: the extremes of the labels
(define (labels-in-range? labels classes)
  (or (zero? (numel labels))
      (let ()
        (match-define (list lo hi) (tensor->list (stack (list (min labels) (max labels)))))
        (and (<= 0 lo) (or (not classes) (< hi classes))))))

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

(define/contract-out (probe-dataset encoder dataset ;; noqa
                                    #:pool [pool 'cls]
                                    #:forward [forward #f]
                                    #:batch-size [batch-size 1024]
                                    #:device [device #f]
                                    #:classes [classes #f])
  (->* ((or/c layer? procedure?) dataset?)
       (#:pool probe-pool/c
        #:forward (or/c #f (procedure-arity-includes/c 1))
        #:batch-size exact-positive-integer?
        #:device (or/c #f device/c)
        #:classes (or/c #f exact-positive-integer?))
       probe-dataset?)
  (error 'unimplemented))

(define/contract-out (tensors->probe-dataset features labels #:classes [classes #f]) ;; noqa
  (->i ([features probe-features/c] [labels labels/c])
       (#:classes [classes (or/c #f exact-positive-integer?)])
       #:pre/name (features labels) "one label per feature row"
       (same-rows? features labels)
       #:pre/name (features labels) "features and labels on one device"
       (same-device? features labels)
       #:pre/name (labels classes) "labels in [0, classes)"
       (labels-in-range? labels (supplied-or classes #f))
       [result probe-dataset?])
  (error 'unimplemented))

(define/contract-out (probe-split ds #:select-per-class [per-class 500] #:seed [seed 0]) ;; noqa
  (->* (probe-dataset?) (#:select-per-class exact-positive-integer? #:seed seed/c)
       (values probe-dataset? probe-dataset?))
  (error 'unimplemented))

(struct linear-probe (weight bias weight-decay sweep mean std classes)
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
       #:pre/name (fit) "at least two fit rows"
       (<= 2 (length fit))
       [result linear-probe?])
  (error 'unimplemented))

(define/contract-out (probe-accuracy probe ds) ;; noqa
  (->i ([probe linear-probe?] [ds probe-dataset?])
       #:pre/name (probe ds) "features of the probe's width, dtype and device"
       (probe-fits? probe ds)
       #:pre/name (probe ds) "classes within the probe's"
       (<= (probe-dataset-classes ds) (linear-probe-classes probe))
       [result (real-in 0 1)])
  (error 'unimplemented))

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
       [result (real-in 0 1)])
  (error 'unimplemented))

(module+ private
  (provide probe-objective)
  (define (probe-objective x y weight bias weight-decay) ;; noqa
    (error 'unimplemented)))
