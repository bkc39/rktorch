#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" item narrow tensor->list)
           (only-in "../nn.rkt"
                    accuracy confusion-matrix metric-compute metric-update! perplexity-metric)
           (only-in "private/python-env.rkt" python-check python-module-available? unpack))

  (define (check-metrics j)
    (define logits (unpack (hash-ref j 'logits) 'float32))
    (define labels (unpack (hash-ref j 'labels) 'int64))
    (check-equal? (hash-ref j 'version) "1.9.0" "M16: the oracle is torchmetrics 1.9.0")
    (for ([k (in-list '(1 3 5))])
      (check-equal? (item (accuracy logits labels #:top-k k #:ignore-index -100))
                    (hash-ref (hash-ref j 'acc) (string->symbol (number->string k)))
                    (format "M16: multiclass_accuracy top_k ~a" k)))
    (check-equal? (item (accuracy logits labels #:ignore-index -100)) (hash-ref j 'plain)
                  "M16: the plain torch formula")
    (check-equal? (tensor->list (confusion-matrix logits labels #:classes 10 #:ignore-index -100))
                  (tensor->list (unpack (hash-ref j 'cm) 'int64))
                  "M16: multiclass_confusion_matrix")
    (define seq (unpack (hash-ref j 'seq) 'float32))
    (define tgt (unpack (hash-ref j 'tgt) 'int64))
    (define ppl (perplexity-metric #:ignore-index -100))
    (metric-update! ppl (narrow seq 0 0 2) (narrow tgt 0 0 2))
    (metric-update! ppl (narrow seq 0 2 2) (narrow tgt 0 2 2))
    (define want (hash-ref j 'ppl))
    (check-true (<= (abs (- (item (metric-compute ppl)) want)) (* 1e-5 want))
                "M16: Perplexity within 1e-5 relative"))

  (cond
    [(not (python-module-available? "torchmetrics"))
     (displayln "[metrics-parity-test] skipped: python3 `torchmetrics` not available")]
    [else (check-metrics (python-check "metrics_twin.py"))]))
