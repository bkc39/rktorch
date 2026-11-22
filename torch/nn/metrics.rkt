#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i any/c contract-out flat-contract-predicate flat-named-contract
                  or/c)
         (only-in racket/list drop-right last)
         (only-in racket/match match-define)
         (only-in "../foreign.rkt"
                  device dtype eq max min numel shape stack tensor->list tensor? where)
         (only-in "../private/contract.rkt" define/contract-out supplied-or))

(define (float-dtype? dt)
  (and (memq dt '(float16 bfloat16 float32 float64)) #t))

(define (rank-of v)
  (length (shape v)))

(define (tensor-of? v ranks dtype?)
  (and (tensor? v) (memv (rank-of v) ranks) (dtype? (dtype v)) #t))

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

(define ignore-index/c (or/c #f exact-integer?))

(define float32-scalar/c
  (flat-named-contract 'float32-scalar (lambda (v) (tensor-of? v '(0) float32?))))

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

(define (rows v)
  (car (shape v)))

(define (classes-of predictions)
  (and (= 2 (rank-of predictions)) (cadr (shape predictions))))

(define (same-rows? predictions labels)
  (= (rows predictions) (rows labels)))

(define (same-device? a b)
  (equal? (device a) (device b)))

(define (top-k-fits? predictions top-k)
  (define classes (classes-of predictions))
  (if classes (<= top-k classes) (= top-k 1)))

(define (width-is? predictions classes)
  (define width (classes-of predictions))
  (or (not width) (= width classes)))

;; one host read: the extremes of t with the ignored entries read as 0
(define (in-range? t bound ignore-index)
  (or (zero? (numel t))
      (let ([kept (if ignore-index (where (eq t ignore-index) 0 t) t)])
        (match-define (list lo hi) (tensor->list (stack (list (min kept) (max kept)))))
        (and (<= 0 lo) (< hi bound)))))

(define (labels-in-range? predictions labels ignore-index)
  (in-range? labels (or (classes-of predictions) +inf.0) ignore-index))

(define (indices-in-range? predictions labels classes ignore-index)
  (and (in-range? labels classes ignore-index)
       (or (classes-of predictions) (in-range? predictions classes #f))))

(define (tokens-match? nll tokens)
  (or (not (tensor? tokens))
      (and (tensor? nll) (same-device? nll tokens))))

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
  (error 'unimplemented))

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
  (error 'unimplemented))

(define/contract-out (perplexity nll #:tokens [tokens #f]) ;; noqa
  (->i ([nll nll/c])
       (#:tokens [tokens tokens/c])
       #:pre/name (nll tokens) "a token tensor needs an NLL tensor on its device"
       (tokens-match? nll (supplied-or tokens #f))
       [result (nll) (if (tensor? nll) float32-scalar/c flonum?)])
  (error 'unimplemented))

(struct metric (kind config [state #:mutable]))

(provide (contract-out [metric? (-> any/c boolean?)]))

(define (config-ref m key)
  (hash-ref (metric-config m) key))

(define (perplexity-metric? m)
  (eq? (metric-kind m) 'perplexity))

(define (metric-predictions/c m)
  (if (perplexity-metric? m) sequence-logits/c predictions/c))

(define (metric-targets/c m)
  (if (perplexity-metric? m) sequence-targets/c labels/c))

(define (rows-match? m predictions targets)
  (if (perplexity-metric? m)
      (equal? (drop-right (shape predictions) 1) (shape targets))
      (same-rows? predictions targets)))

(define (width-fits? m predictions)
  (case (metric-kind m)
    [(accuracy) (top-k-fits? predictions (config-ref m 'top-k))]
    [(confusion-matrix) (width-is? predictions (config-ref m 'classes))]
    [else #t]))

(define (device-fits? m predictions)
  (define state (metric-state m))
  (or (not state) (same-device? (car state) predictions)))

(define (range-fits? m predictions targets)
  (define ignore-index (config-ref m 'ignore-index))
  (or (not (config-ref m 'validate-args?))
      (case (metric-kind m)
        [(accuracy) (labels-in-range? predictions targets ignore-index)]
        [(confusion-matrix)
         (indices-in-range? predictions targets (config-ref m 'classes) ignore-index)]
        [else (in-range? targets (last (shape predictions)) ignore-index)])))

(define (metric-updated? m)
  (and (metric-state m) #t))

(define/contract-out (accuracy-metric #:top-k [top-k 1] ;; noqa
                                      #:ignore-index [ignore-index #f]
                                      #:validate-args? [validate-args? #t])
  (->* () (#:top-k exact-positive-integer? #:ignore-index ignore-index/c #:validate-args? boolean?)
       metric?)
  (error 'unimplemented))

(define/contract-out (confusion-matrix-metric #:classes classes ;; noqa
                                              #:ignore-index [ignore-index #f]
                                              #:validate-args? [validate-args? #t])
  (->* (#:classes exact-positive-integer?)
       (#:ignore-index ignore-index/c #:validate-args? boolean?)
       metric?)
  (error 'unimplemented))

(define/contract-out (perplexity-metric #:ignore-index [ignore-index #f] ;; noqa
                                        #:validate-args? [validate-args? #t])
  (->* () (#:ignore-index ignore-index/c #:validate-args? boolean?) metric?)
  (error 'unimplemented))

(define/contract-out (metric-update! m predictions targets) ;; noqa
  (->i ([m metric?]
        [predictions (m) (metric-predictions/c m)]
        [targets (m) (metric-targets/c m)])
       #:pre/name (m predictions targets) "one target per prediction row"
       (rows-match? m predictions targets)
       #:pre/name (m predictions) "logits of the metric's class count, top-k within it"
       (width-fits? m predictions)
       #:pre/name (predictions targets) "predictions and targets on one device"
       (same-device? predictions targets)
       #:pre/name (m predictions) "the device of this metric's earlier updates"
       (device-fits? m predictions)
       #:pre/name (m predictions targets) "targets in range, or the ignore index"
       (range-fits? m predictions targets)
       [result void?])
  (error 'unimplemented))

(define/contract-out (metric-compute m) ;; noqa
  (->i ([m metric?])
       #:pre/name (m) "updated since construction or the last reset"
       (metric-updated? m)
       [result tensor?])
  (error 'unimplemented))

(define/contract-out (metric-reset! m) ;; noqa
  (-> metric? void?)
  (error 'unimplemented))
