#lang racket/base

(require (only-in racket/list append* append-map count)
         (only-in racket/match match match-define match-lambda)
         (only-in racket/string string-prefix? string-replace)
         (only-in rackunit check-= check-equal?)
         (only-in torch ~> item reshape tensor->list)
         (only-in torch/nn
                  clip-grad-norm! grad-scaler-scale learning-rate optimizer-param-groups
                  param-group-count param-group-params)
         (only-in torch/train
                  gen:training-logger lambda-callback run-grad-scaler))

(provide (struct-out trace)
         trace-recorder
         trace-callback
         trace-logger
         lightning-name
         lightning-row
         flat-state
         compare-trace
         trace-gaps)

(struct trace ([labels #:mutable] [losses #:mutable] [lrs #:mutable] [norms #:mutable]
               [scales #:mutable] [rows #:mutable]))

(define (trace-recorder)
  (trace '() '() '() '() '() '()))

(define (trace-callback tr)
  (define (batch-end! _r loss _extra _xb yb)
    (set-trace-labels! tr (cons (tensor->list yb) (trace-labels tr)))
    (set-trace-losses! tr (cons (item loss) (trace-losses tr))))
  (define (before-step! r opt)
    (define params (append-map param-group-params (optimizer-param-groups opt)))
    (set-trace-lrs! tr (cons (for/list ([g (in-range (param-group-count opt))])
                               (learning-rate opt #:group g))
                             (trace-lrs tr)))
    (set-trace-norms! tr (cons (item (clip-grad-norm! params +inf.0)) (trace-norms tr)))
    (match (run-grad-scaler r)
      [#f (void)]
      [s (set-trace-scales! tr (cons (grad-scaler-scale s) (trace-scales tr)))]))
  (lambda-callback #:on-train-batch-end batch-end! #:on-before-optimizer-step before-step!))

(struct row-logger (tr)
  #:methods gen:training-logger
  [(define (logger-log-hyperparams! l _params) (void))
   (define (logger-log-metrics! l metrics step)
     (define tr (row-logger-tr l))
     (set-trace-rows! tr (cons (list step metrics) (trace-rows tr))))
   (define (logger-finalize! l _status) (void))])

(define (trace-logger tr)
  (row-logger tr))

(define (lightning-name s)
  (string-replace s "-" "_"))

(define (lightning-row row)
  (match-define (list step metrics) row)
  (list step
        (for/hash ([(k v) (in-hash metrics)])
          (values (lightning-name (symbol->string k)) (exact->inexact v)))))

(define (json-row row)
  (match-define (list step metrics) row)
  (list step (for/hash ([(k v) (in-hash metrics)]) (values (symbol->string k) v))))

(define (flat-state named)
  (for/list ([np (in-list named)])
    (match-define (cons name t) np)
    (cons (lightning-name name) (~> t (reshape -1) tensor->list))))

(define (check-close label xs ys tolerance)
  (check-equal? (length xs) (length ys) (format "~a: count" label))
  (for ([x (in-list xs)] [y (in-list ys)] [i (in-naturals)])
    (check-= x y tolerance (format "~a[~a]" label i))))

(define (val-row? row)
  (for/or ([k (in-hash-keys (cadr row))]) (string-prefix? k "val/")))

(define (row-keys rows)
  (map (match-lambda [(list _ m) (sort (hash-keys m) string<?)]) rows))

(define (check-rows label ours theirs epochs tolerance acc-tolerance)
  (check-equal? (map car ours) (map car theirs) (format "~a: row steps" label))
  (check-equal? (row-keys ours) (row-keys theirs) (format "~a: row keys" label))
  (check-equal? (count val-row? ours) epochs (format "~a: validation rows" label))
  (check-equal? (count val-row? theirs) epochs (format "~a: twin validation rows" label))
  (for ([o (in-list ours)] [t (in-list theirs)] [i (in-naturals)])
    (match-define (list _ om) o)
    (for ([(k v) (in-hash (cadr t))] #:when (hash-has-key? om k))
      (check-= (hash-ref om k) v (if (equal? k "val/acc") acc-tolerance tolerance)
               (format "~a: row ~a ~a" label i k)))))

(define (check-state label state json tolerance)
  (for ([entry (in-list state)])
    (match-define (cons name values) entry)
    (define theirs (hash-ref json (string->symbol name)))
    (cond
      [(regexp-match? #rx"num_batches_tracked$" name)
       (check-equal? values theirs (format "~a: ~a" label name))]
      [else (check-close (format "~a: ~a" label name) values theirs tolerance)])))

(define (compare-trace label tr json state tolerance #:acc-tolerance [acc-tolerance tolerance])
  (define epochs (hash-ref (hash-ref json 'config) 'epochs))
  (check-equal? (reverse (trace-labels tr)) (hash-ref json 'labels) (format "~a: labels" label))
  (check-close (format "~a: losses" label) (reverse (trace-losses tr)) (hash-ref json 'losses)
               tolerance)
  (check-equal? (reverse (trace-lrs tr)) (hash-ref json 'lrs) (format "~a: lrs" label))
  (check-close (format "~a: norms" label) (reverse (trace-norms tr)) (hash-ref json 'norms)
               tolerance)
  (check-equal? (reverse (trace-scales tr)) (hash-ref json 'scales) (format "~a: scales" label))
  (check-rows label
              (map lightning-row (reverse (trace-rows tr)))
              (map json-row (hash-ref json 'rows))
              epochs tolerance acc-tolerance)
  (check-state label state (hash-ref json 'state) tolerance))

(define (largest-gap xs ys)
  (for/fold ([gap 0.0]) ([x (in-list xs)] [y (in-list ys)])
    (max gap (abs (- x y)))))

(define (trace-gaps tr json state)
  (define theirs (hash-ref json 'state))
  (define (row-values rows)
    (for*/list ([row (in-list rows)] [k (in-list (sort (hash-keys (cadr row)) string<?))])
      (hash-ref (cadr row) k)))
  (hasheq 'losses (largest-gap (reverse (trace-losses tr)) (hash-ref json 'losses))
          'lrs (largest-gap (append* (reverse (trace-lrs tr))) (append* (hash-ref json 'lrs)))
          'norms (largest-gap (reverse (trace-norms tr)) (hash-ref json 'norms))
          'scales (largest-gap (reverse (trace-scales tr)) (hash-ref json 'scales))
          'rows (largest-gap (row-values (map lightning-row (reverse (trace-rows tr))))
                             (row-values (map json-row (hash-ref json 'rows))))
          'state (for/fold ([gap 0.0]) ([entry (in-list state)])
                   (max gap (largest-gap (cdr entry)
                                         (hash-ref theirs (string->symbol (car entry))))))))
