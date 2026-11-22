#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i any/c contract-out flat-contract-predicate flat-named-contract
                  integer-in or/c)
         (only-in racket/flonum fl/ flexp)
         (only-in racket/generic define-generics)
         (only-in racket/list drop-right last)
         (only-in racket/match match match-define)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  * + / argmax copy! device dtype eq exp full index-add! log-softmax ne ones-like
                  reshape shape sum tensor? to-dtype topk unsqueeze with-no-grad zeros)
         (prefix-in g: (only-in "../generated.rkt" nll-loss sum-dim-intlist))
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "private/tensor-checks.rkt"
                  in-range? rank-of same-device? same-rows? tensor-of?))

(provide (contract-out [metric? (-> any/c boolean?)]))

(module+ private
  (provide tally-state))

(define (float-dtype? dt)
  (and (memq dt '(float16 bfloat16 float32 float64)) #t))

(define (int64? dt)
  (eq? dt 'int64))

(define (float32? dt)
  (eq? dt 'float32))

(define logits/c
  (flat-named-contract 'logits (lambda (v) (tensor-of? v '(2) float-dtype?))))

(define class-indices/c
  (flat-named-contract 'class-indices (lambda (v) (tensor-of? v '(1) int64?))))

(define predictions/c (or/c logits/c class-indices/c))

(define labels/c (flat-named-contract 'labels (flat-contract-predicate class-indices/c)))

(define int64-min (- (expt 2 63)))

(define ignore-index/c (or/c #f (integer-in int64-min (sub1 (expt 2 63)))))

(define float32-scalar/c
  (flat-named-contract 'float32-scalar (lambda (v) (tensor-of? v '(0) float32?))))

(define (scalar-of/c dt)
  (flat-named-contract (string->symbol (format "~a-scalar" dt))
                       (lambda (v) (tensor-of? v '(0) (lambda (d) (eq? d dt))))))

(define (confusion/c classes)
  (flat-named-contract 'confusion-matrix
                       (lambda (v)
                         (and (tensor-of? v '(2) int64?) (equal? (shape v) (list classes classes))))))

(define nll/c
  (or/c real? (flat-named-contract 'float-scalar-tensor
                                   (lambda (v) (tensor-of? v '(0) float-dtype?)))))

(define tokens/c
  (or/c #f exact-positive-integer?
        (flat-named-contract 'int64-scalar-tensor (lambda (v) (tensor-of? v '(0) int64?)))))

(define sequence-logits/c
  (flat-named-contract 'sequence-logits (lambda (v) (tensor-of? v '(2 3) float-dtype?))))

(define sequence-targets/c
  (flat-named-contract 'sequence-targets (lambda (v) (tensor-of? v '(1 2) int64?))))

(define (classes-of predictions)
  (match (shape predictions)
    [(list _ classes) classes]
    [_ #f]))

(define (top-k-fits? predictions top-k)
  (define classes (classes-of predictions))
  (if classes (<= top-k classes) (= top-k 1)))

(define (width-is? predictions classes)
  (define width (classes-of predictions))
  (or (not width) (= width classes)))

(define (labels-in-range? predictions labels ignore-index)
  (in-range? labels (or (classes-of predictions) +inf.0) ignore-index))

(define (indices-in-range? predictions labels classes ignore-index)
  (and (in-range? labels classes ignore-index)
       (or (classes-of predictions) (in-range? predictions classes))))

(define (tokens-match? nll tokens)
  (or (not (tensor? tokens))
      (and (tensor? nll) (same-device? nll tokens))))

(define (predicted-classes predictions)
  (if (classes-of predictions) (argmax predictions 1) predictions))

(define (row-hits predictions labels top-k)
  (cond
    [(= top-k 1) (~> (predicted-classes predictions) (eq labels) (to-dtype 'int64))]
    [else
     (define-values (_top indices) (topk predictions top-k #:dim 1))
     (g:sum-dim-intlist (eq indices (unsqueeze labels 1)) '(1) #f 'int64)]))

(define (int64-scalar v on)
  (full v #:dtype 'int64 #:device on))

(define (kept targets ignore-index)
  (ne targets (int64-scalar ignore-index (device targets))))

(define (counted labels ignore-index)
  (if ignore-index (to-dtype (kept labels ignore-index) 'int64) (ones-like labels)))

(define (accuracy-counts predictions labels top-k ignore-index)
  (define weights (counted labels ignore-index))
  (list (~> (row-hits predictions labels top-k) (* weights) sum) (sum weights)))

(define (ratio counts)
  (match-define (list hits total) counts)
  (/ (to-dtype hits 'float32) (to-dtype total 'float32)))

;; an ignored row adds its zero weight at cell 0, whatever its label
(define (confusion-add! counts predictions labels classes ignore-index)
  (define weights (counted labels ignore-index))
  (define cells (+ (* labels (int64-scalar classes (device labels))) (predicted-classes predictions)))
  (index-add! counts 0 (if ignore-index (* cells weights) cells) weights))

(define (fresh-counts classes on)
  (zeros (* classes classes) #:dtype 'int64 #:device on))

(define sum-reduction 2)

(define (perplexity-counts logits targets ignore-index)
  ;; no target in [0, V) is int64-min, so #f ignores none
  (define ignored (or ignore-index int64-min))
  (define flat-targets (reshape targets -1))
  (define nll
    (~> (reshape logits -1 (last (shape logits)))
        (log-softmax -1)
        (g:nll-loss flat-targets #f sum-reduction ignored)))
  (list (to-dtype nll 'float32) (~> (kept flat-targets ignored) (to-dtype 'float32) sum)))

(define/contract-out (accuracy predictions labels ;; noqa
                               #:top-k [top-k 1]
                               #:ignore-index [ignore-index #f]
                               #:validate-args? [validate-args? #t])
  (->i ([predictions predictions/c] [labels labels/c])
       (#:top-k [top-k exact-positive-integer?]
        #:ignore-index [ignore-index ignore-index/c]
        #:validate-args? [validate-args? boolean?])
       #:pre/name (predictions labels) "one label per prediction row"
       (same-rows? predictions labels)
       #:pre/name (predictions labels) "predictions and labels on one device"
       (same-device? predictions labels)
       #:pre/name (predictions top-k) "top-k at most the class count, and 1 for class indices"
       (top-k-fits? predictions (supplied-or top-k 1))
       #:pre/name (predictions labels ignore-index validate-args?)
       "labels in [0, classes) or the ignore index"
       (or (not (supplied-or validate-args? #t))
           (labels-in-range? predictions labels (supplied-or ignore-index #f)))
       [result float32-scalar/c])
  (with-no-grad (ratio (accuracy-counts predictions labels top-k ignore-index))))

(define/contract-out (confusion-matrix predictions labels ;; noqa
                                       #:classes classes
                                       #:ignore-index [ignore-index #f]
                                       #:validate-args? [validate-args? #t])
  (->i ([predictions predictions/c] [labels labels/c] #:classes [classes exact-positive-integer?])
       (#:ignore-index [ignore-index ignore-index/c] #:validate-args? [validate-args? boolean?])
       #:pre/name (predictions labels) "one label per prediction row"
       (same-rows? predictions labels)
       #:pre/name (predictions labels) "predictions and labels on one device"
       (same-device? predictions labels)
       #:pre/name (predictions classes) "logits with one column per class"
       (width-is? predictions classes)
       #:pre/name (predictions labels classes ignore-index validate-args?)
       "labels and predicted classes in [0, classes), labels possibly the ignore index"
       (or (not (supplied-or validate-args? #t))
           (indices-in-range? predictions labels classes (supplied-or ignore-index #f)))
       [result (classes) (confusion/c classes)])
  (with-no-grad
    (define counts (fresh-counts classes (device labels)))
    (confusion-add! counts predictions labels classes ignore-index)
    (reshape counts classes classes)))

(define/contract-out (perplexity nll #:tokens [tokens #f]) ;; noqa
  (->i ([nll nll/c])
       (#:tokens [tokens tokens/c])
       #:pre/name (nll tokens) "a token tensor needs an NLL tensor on its device"
       (tokens-match? nll (supplied-or tokens #f))
       [result (nll) (if (tensor? nll) (scalar-of/c (dtype nll)) flonum?)])
  (if (tensor? nll)
      (exp (if tokens (/ nll tokens) nll))
      (flexp (if tokens (fl/ (real->double-flonum nll) (real->double-flonum tokens))
                 (real->double-flonum nll)))))

(define-generics metric
  (predictions-of metric)
  (targets-of metric)
  (rows-fit? metric predictions targets)
  (width-fits? metric predictions)
  (targets-in-range? metric predictions targets)
  (tally-add metric state predictions targets)
  (tally-figure metric state)
  #:fallbacks
  [(define (predictions-of _m) predictions/c)
   (define (targets-of _m) labels/c)
   (define (rows-fit? _m predictions labels) (same-rows? predictions labels))])

(struct tally (ignore-index validate-args? [state #:mutable]))

(define (summed sums counts)
  (if sums (map + sums counts) counts))

(struct accuracy-tally tally (top-k)
  #:methods gen:metric
  [(define (width-fits? m predictions)
     (top-k-fits? predictions (accuracy-tally-top-k m)))
   (define (targets-in-range? m predictions labels)
     (labels-in-range? predictions labels (tally-ignore-index m)))
   (define (tally-add m sums predictions labels)
     (summed sums (accuracy-counts predictions labels (accuracy-tally-top-k m)
                                   (tally-ignore-index m))))
   (define (tally-figure _m sums)
     (ratio sums))])

(struct confusion-tally tally (classes)
  #:methods gen:metric
  [(define (width-fits? m predictions)
     (width-is? predictions (confusion-tally-classes m)))
   (define (targets-in-range? m predictions labels)
     (indices-in-range? predictions labels (confusion-tally-classes m) (tally-ignore-index m)))
   (define (tally-add m counts predictions labels)
     (define classes (confusion-tally-classes m))
     (define into (or counts (fresh-counts classes (device labels))))
     (confusion-add! into predictions labels classes (tally-ignore-index m))
     into)
   (define (tally-figure m counts)
     (define classes (confusion-tally-classes m))
     (define fresh (zeros classes classes #:dtype 'int64 #:device (device counts)))
     (copy! fresh (reshape counts classes classes))
     fresh)])

(struct perplexity-tally tally ()
  #:methods gen:metric
  [(define (predictions-of _m) sequence-logits/c)
   (define (targets-of _m) sequence-targets/c)
   (define (rows-fit? _m logits targets)
     (equal? (drop-right (shape logits) 1) (shape targets)))
   (define (width-fits? _m _logits) #t)
   (define (targets-in-range? m logits targets)
     (in-range? targets (last (shape logits)) (tally-ignore-index m)))
   (define (tally-add m sums logits targets)
     (summed sums (perplexity-counts logits targets (tally-ignore-index m))))
   (define (tally-figure _m sums)
     (match-define (list nll tokens) sums)
     (exp (/ nll tokens)))])

(define (device-fits? m predictions)
  (match (tally-state m)
    [#f #t]
    [(or (list on _) on) (same-device? on predictions)]))

(define (range-fits? m predictions targets)
  (or (not (tally-validate-args? m)) (targets-in-range? m predictions targets)))

(define (metric-updated? m)
  (and (tally-state m) #t))

(define/contract-out (accuracy-metric #:top-k [top-k 1] ;; noqa
                                      #:ignore-index [ignore-index #f]
                                      #:validate-args? [validate-args? #t])
  (->* [] [#:top-k exact-positive-integer? #:ignore-index ignore-index/c #:validate-args? boolean?]
       metric?)
  (accuracy-tally ignore-index validate-args? #f top-k))

(define/contract-out (confusion-matrix-metric #:classes classes ;; noqa
                                              #:ignore-index [ignore-index #f]
                                              #:validate-args? [validate-args? #t])
  (->* [#:classes exact-positive-integer?]
       [#:ignore-index ignore-index/c #:validate-args? boolean?]
       metric?)
  (confusion-tally ignore-index validate-args? #f classes))

(define/contract-out (perplexity-metric #:ignore-index [ignore-index #f] ;; noqa
                                        #:validate-args? [validate-args? #t])
  (->* [] [#:ignore-index ignore-index/c #:validate-args? boolean?] metric?)
  (perplexity-tally ignore-index validate-args? #f))

(define/contract-out (metric-update! m predictions targets) ;; noqa
  (->i ([m metric?]
        [predictions (m) (predictions-of m)]
        [targets (m) (targets-of m)])
       #:pre/name (m predictions targets) "one target per prediction row"
       (rows-fit? m predictions targets)
       #:pre/name (m predictions) "logits of the metric's class count, top-k within it"
       (width-fits? m predictions)
       #:pre/name (predictions targets) "predictions and targets on one device"
       (same-device? predictions targets)
       #:pre/name (m predictions) "the device of this metric's earlier updates"
       (device-fits? m predictions)
       #:pre/name (m predictions targets) "targets in range, or the ignore index"
       (range-fits? m predictions targets)
       [result void?])
  (with-no-grad
    (set-tally-state! m (tally-add m (tally-state m) predictions targets))))

(define/contract-out (metric-compute m) ;; noqa
  (->i ([m metric?])
       #:pre/name (m) "updated since construction or the last reset"
       (metric-updated? m)
       [result tensor?])
  (with-no-grad (tally-figure m (tally-state m))))

(define/contract-out (metric-reset! m) ;; noqa
  (-> metric? void?)
  (set-tally-state! m #f))
