#lang racket/base

(require racket/contract/base
         (only-in racket/list append* append-map count first second take take-right)
         (only-in racket/match == match-define match-lambda)
         (only-in racket/math infinite? nan?)
         (only-in racket/string string-prefix? string-replace)
         (only-in rackunit check-equal? check-true)
         (only-in torch ~> item reshape shape tensor->list tensor?)
         (only-in torch/nn
                  clip-grad-norm! grad-scaler-scale learning-rate optimizer-param-groups
                  param-group-count param-group-params)
         (only-in torch/private/contract define/contract-out)
         (only-in torch/train
                  gen:training-logger lambda-callback run-grad-scaler))

(provide trace-recorder
         trace-callback
         trace-logger
         trace-labels
         trace-losses
         trace-lrs
         trace-norms
         trace-scales
         trace-rows
         lightning-name
         flat-state
         check-close
         largest-gap
         compare-trace
         trace-gaps)

(struct trace-recorder ([events #:auto #:mutable]) #:auto-value '())

(define (note! tr kind v)
  (set-trace-recorder-events! tr (cons (cons kind v) (trace-recorder-events tr))))

(define ((events-of kind) tr)
  (append-map (match-lambda
                [(cons (== kind eq?) v) (list v)]
                [_ '()])
              (reverse (trace-recorder-events tr))))

(define trace-labels (events-of 'label))
(define trace-losses (events-of 'loss))
(define trace-lrs (events-of 'lrs))
(define trace-norms (events-of 'norm))
(define trace-scales (events-of 'scale))
(define trace-rows (events-of 'row))

(define (trace-callback tr)
  (define (batch-end! _r loss _extra _xb yb)
    (note! tr 'label (tensor->list yb))
    (note! tr 'loss (item loss)))
  (define (before-step! r opt)
    (define params (append-map param-group-params (optimizer-param-groups opt)))
    (note! tr 'lrs (for/list ([g (in-range (param-group-count opt))])
                     (learning-rate opt #:group g)))
    (note! tr 'norm (item (clip-grad-norm! params +inf.0)))
    (define scaler (run-grad-scaler r))
    (when scaler
      (note! tr 'scale (grad-scaler-scale scaler))))
  (lambda-callback #:on-train-batch-end batch-end! #:on-before-optimizer-step before-step!))

(struct trace-logger (tr)
  #:methods gen:training-logger
  [(define (logger-log-hyperparams! l _params) (void))
   (define (logger-log-metrics! l metrics step)
     (note! (trace-logger-tr l) 'row (list step metrics)))
   (define (logger-finalize! l _status) (void))])

(define (lightning-name s)
  (string-replace s "-" "_"))

(define (comparable x)
  (cond
    [(not (real? x)) x]
    [(nan? x) "nan"]
    [(infinite? x) (if (positive? x) "inf" "-inf")]
    [else (exact->inexact x)]))

(define (gap x y)
  (match-define (list a b) (map comparable (list x y)))
  (cond
    [(and (real? a) (real? b)) (abs (- a b))]
    [(equal? a b) 0.0]
    [else +inf.0]))

(define (gaps xs ys)
  (for/list ([x (in-list xs)] [y (in-list ys)])
    (gap x y)))

(define (largest-gap xs ys)
  (apply max 0.0 (gaps xs ys)))

(define (check-close label xs ys tolerance)
  (check-equal? (length xs) (length ys) (format "~a: count" label))
  (for ([g (in-list (gaps xs ys))] [x (in-list xs)] [y (in-list ys)] [i (in-naturals)])
    (check-true (<= g tolerance) (format "~a[~a]: ~a against ~a" label i x y))))

(define ((row-by key->name) row)
  (match-define (list step metrics) row)
  (list step (for/hash ([(k v) (in-hash metrics)]) (values (key->name k) (comparable v)))))

(define lightning-row (row-by (lambda (k) (lightning-name (symbol->string k)))))
(define json-row (row-by symbol->string))

(define (flat-state named)
  (for/list ([np (in-list named)])
    (match-define (cons name t) np)
    (cons (lightning-name name) (~> t (reshape -1) tensor->list))))

(define (sorted-keys metrics)
  (sort (hash-keys metrics) string<?))

(define val-row?
  (match-lambda [(list _ m) (for/or ([k (in-hash-keys m)]) (string-prefix? k "val/"))]))

(define (row-steps rows)
  (map (match-lambda [(list step _) step]) rows))

(define (row-keys rows)
  (map (match-lambda [(list _ m) (sorted-keys m)]) rows))

(define (row-values rows)
  (append-map (match-lambda [(list _ m) (map (lambda (k) (hash-ref m k)) (sorted-keys m))])
              rows))

(define (check-state label state json tolerance)
  (for ([entry (in-list state)])
    (match-define (cons name vals) entry)
    (define theirs (hash-ref json (string->symbol name)))
    (cond
      [(regexp-match? #rx"num_batches_tracked$" name)
       (check-equal? vals theirs (format "~a: ~a" label name))]
      [else (check-close (format "~a: ~a" label name) vals theirs tolerance)])))

(define (check-steps label tr json)
  (define epochs (hash-ref (hash-ref json 'config) 'epochs))
  (define ours (map lightning-row (trace-rows tr)))
  (define theirs (map json-row (hash-ref json 'rows)))
  (check-equal? (trace-labels tr) (hash-ref json 'labels) (format "~a: labels" label))
  (check-equal? (trace-lrs tr) (hash-ref json 'lrs) (format "~a: lrs" label))
  (check-equal? (trace-scales tr) (hash-ref json 'scales) (format "~a: scales" label))
  (check-equal? (row-steps ours) (row-steps theirs) (format "~a: row steps" label))
  (check-equal? (row-keys ours) (row-keys theirs) (format "~a: row keys" label))
  (check-equal? (count val-row? ours) epochs (format "~a: validation rows" label))
  (check-equal? (count val-row? theirs) epochs (format "~a: twin validation rows" label)))

(define (check-rows label ours theirs tolerance acc-tolerance)
  (for ([o (in-list ours)] [t (in-list theirs)] [i (in-naturals)])
    (match-define (list _ om) o)
    (match-define (list _ tm) t)
    (for ([(k v) (in-hash tm)] #:when (hash-has-key? om k))
      (check-close (format "~a: row ~a ~a" label i k) (list (hash-ref om k)) (list v)
                   (if (equal? k "val/acc") acc-tolerance tolerance)))))

(define (compare-trace label tr json state tolerance #:acc-tolerance [acc-tolerance tolerance])
  (check-steps label tr json)
  (check-close (format "~a: losses" label) (trace-losses tr) (hash-ref json 'losses) tolerance)
  (check-close (format "~a: norms" label) (trace-norms tr) (hash-ref json 'norms) tolerance)
  (check-rows label
              (map lightning-row (trace-rows tr))
              (map json-row (hash-ref json 'rows))
              tolerance acc-tolerance)
  (check-state label state (hash-ref json 'state) tolerance))

(define (trace-gaps tr json state)
  (define theirs (hash-ref json 'state))
  (hasheq 'losses (largest-gap (trace-losses tr) (hash-ref json 'losses))
          'lrs (largest-gap (append* (trace-lrs tr)) (append* (hash-ref json 'lrs)))
          'norms (largest-gap (trace-norms tr) (hash-ref json 'norms))
          'scales (largest-gap (trace-scales tr) (hash-ref json 'scales))
          'rows (largest-gap (row-values (map lightning-row (trace-rows tr)))
                             (row-values (map json-row (hash-ref json 'rows))))
          'state (for/fold ([widest 0.0]) ([entry (in-list state)])
                   (match-define (cons name vals) entry)
                   (max widest (largest-gap vals (hash-ref theirs (string->symbol name)))))))

(define (tensor-summary t)
  (define vals (~> t (reshape -1) tensor->list))
  (define n (min 8 (length vals)))
  (hasheq 'shape (shape t)
          'sum (for/sum ([v (in-list vals)]) v)
          'sum-sq (for/sum ([v (in-list vals)]) (* v v))
          'head (take vals n)
          'tail (take-right vals n)))

(define/contract-out (state-summary named)
  (-> (listof (cons/c string? tensor?)) (and/c hash? hash-eq?))
  (for/hasheq ([np (in-list named)])
    (match-define (cons name t) np)
    (values (string->symbol (lightning-name name)) (tensor-summary t))))

(define golden-quantities '(losses norms rows acc values sum sum-sq))

(define (counter? name)
  (regexp-match? #rx"num_batches_tracked$" (symbol->string name)))

(define (nonzero-or-1 x)
  (if (and (real? x) (positive? x)) x 1.0))

(define (entry-gap e)
  (match-define (list _ _ x y scale) e)
  (/ (gap x y) scale))

(define (series-entries quantity xs ys)
  (for/list ([x (in-list xs)] [y (in-list ys)] [i (in-naturals)])
    (list quantity (format "~a[~a]" quantity i) x y 1.0)))

(define (row-entries ours theirs)
  (for*/list ([(o t) (in-parallel (in-list ours) (in-list theirs))]
              [(k v) (in-hash (second t))])
    (list (if (equal? k "val/acc") 'acc 'rows) (format "row ~a ~a" (first t) k)
          (hash-ref (second o) k #f) v 1.0)))

(define (tensor-entries o t)
  (match-define (hash-table ('shape dims) ('sum-sq sum-sq)) t)
  (define n (apply * dims))
  (append
   (list (list 'sum 'sum (hash-ref o 'sum #f) (hash-ref t 'sum)
               (~> (* n sum-sq) sqrt nonzero-or-1))
         (list 'sum-sq 'sum-sq (hash-ref o 'sum-sq #f) sum-sq (nonzero-or-1 sum-sq)))
   (for*/list ([k (in-list '(head tail))]
               [(x y) (in-parallel (in-list (hash-ref o k '())) (in-list (hash-ref t k)))])
     (list 'values k x y 1.0))))

(define (summary-entries kind ours theirs)
  (for*/list ([name (in-list (sort (hash-keys theirs) symbol<?))]
              #:unless (counter? name)
              [e (in-list (tensor-entries (hash-ref ours name (hasheq)) (hash-ref theirs name)))])
    (match-define (list quantity k x y scale) e)
    (list quantity (format "~a ~a ~a" kind name k) x y scale)))

(define (golden-entries tr golden init state)
  (append (series-entries 'losses (trace-losses tr) (hash-ref golden 'losses))
          (series-entries 'norms (trace-norms tr) (hash-ref golden 'norms))
          (row-entries (map lightning-row (trace-rows tr)) (map json-row (hash-ref golden 'rows)))
          (summary-entries "init" init (hash-ref golden 'init))
          (summary-entries "state" state (hash-ref golden 'state))))

(define (check-summary-layout label kind ours theirs)
  (check-equal? (sort (hash-keys ours) symbol<?) (sort (hash-keys theirs) symbol<?)
                (format "~a: ~a names" label kind))
  (for ([(name t) (in-hash theirs)])
    (define o (hash-ref ours name (hasheq)))
    (check-equal? (hash-ref o 'shape #f) (hash-ref t 'shape)
                  (format "~a: ~a ~a shape" label kind name))
    (when (counter? name)
      (check-equal? o t (format "~a: ~a ~a" label kind name)))))

(define/contract-out (compare-golden label tr golden init state tolerances)
  (-> string? trace-recorder? hash? hash? hash? (hash/c symbol? (>=/c 0)) void?)
  (check-steps label tr golden)
  (check-equal? (length (trace-losses tr)) (length (hash-ref golden 'losses))
                (format "~a: losses count" label))
  (check-equal? (length (trace-norms tr)) (length (hash-ref golden 'norms))
                (format "~a: norms count" label))
  (check-summary-layout label "init" init (hash-ref golden 'init))
  (check-summary-layout label "state" state (hash-ref golden 'state))
  (for ([e (in-list (golden-entries tr golden init state))])
    (match-define (list quantity where x y _) e)
    (check-true (<= (entry-gap e) (hash-ref tolerances quantity))
                (format "~a: ~a: ~a against ~a" label where x y))))

(define/contract-out (golden-gaps tr golden init state)
  (-> trace-recorder? hash? hash? hash? (hash/c symbol? real?))
  (for/fold ([widest (for/hasheq ([q (in-list golden-quantities)]) (values q 0.0))])
            ([e (in-list (golden-entries tr golden init state))])
    (hash-update widest (first e) (lambda (g) (max g (entry-gap e))))))

(module+ test
  (require (only-in json string->jsexpr)
           (only-in rackunit test-case))

  (test-case "a non-finite value matches the twin's sentinel, and only it"
    (define theirs (string->jsexpr "[\"inf\", 1.0, \"nan\", \"-inf\"]"))
    (check-close "norms" (list +inf.0 1.0 +nan.0 -inf.0) theirs 0.0)
    (check-equal? (largest-gap (list +inf.0 1.0 +nan.0 -inf.0) theirs) 0.0)
    (check-equal? (largest-gap (list +inf.0) (list "-inf")) +inf.0)
    (check-equal? (largest-gap (list 2.0) (list "inf")) +inf.0)
    (check-equal? (largest-gap (list 1 2.5) (list 1.0 2.0)) 0.5)))
