#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-contains?)
         rackunit
         (only-in "markdown.rkt" ref)
         (only-in "summary-op-cost.rkt"
                  attribution crossing-us gap-tolerance k1 k2 k3 k4 k5 k6 layer-median
                  op-cost-markdown paired-delta piece-label read-records samples-by-rep))

(define-runtime-path fixture "fixtures/op-cost/records.jsonl")

(define records (read-records fixture))

(define-check (check-near actual expected)
  (unless (if (list? expected)
              (and (list? actual) (= (length actual) (length expected))
                   (andmap (lambda (a e) (< (abs (- a e)) 1e-9)) actual expected))
              (and (real? actual) (< (abs (- actual expected)) 1e-9)))
    (fail-check)))

(define (piece-of a layer)
  (for/first ([p (in-list (ref a 'pieces))] #:when (equal? (ref p 'layer) layer)) p))

(test-case "the records file parses every line, and a blank line is no record"
  (check-equal? (length records) 68))

(test-case "a condition's samples become one median per repetition"
  (define rows
    (list (hasheq 'rep 0 'us_per_op 1.0) (hasheq 'rep 0 'us_per_op 3.0)
          (hasheq 'rep 1 'us_per_op 5.0) (hasheq 'rep 2 'us_per_op #f)))
  (check-equal? (samples-by-rep rows 'us_per_op) (hash 0 2.0 1 5.0)))

(test-case "a paired delta takes the median of per-rep differences, over the reps both have"
  (check-near (paired-delta (hash 0 10 1 20 2 30 3 99) (hash 0 9 1 18 2 27)) 2)
  (check-false (paired-delta (hash 0 1) (hash 1 1))))

(test-case "a piece spanning a missing layer is labelled with both"
  (check-equal? (piece-label '(retry accounted))
                "oom-retry wrapper + accounting (placement, record-allocation!)"))

(test-case "the pieces of a ladder are the paired steps between the layers present"
  (define a (attribution records "scalar-add"))
  (check-equal? (map (lambda (p) (ref p 'layer)) (ref a 'pieces))
                '("raw" "registered" "accounted" "checked" "wrapped" "unchecked" "facade"))
  (check-near (map (lambda (p) (ref p 'us)) (ref a 'pieces)) '(1 2 3 4 2 1 1))
  (check-equal? (ref (piece-of a "accounted") 'piece)
                "oom-retry wrapper + accounting (placement, record-allocation!)")
  (check-near (ref (piece-of a "registered") 'heap) 1000)
  (check-equal? (ref (piece-of a "raw") 'crossings) 2)
  (check-equal? (ref (piece-of a "wrapped") 'crossings) 2)
  (check-near (ref a 'contract) 5))

(test-case "the pieces sum against the total measured on its own, and the gap is reported"
  (define a (attribution records "scalar-add"))
  (check-near (ref a 'sum) 14)
  (check-near (ref a 'total) 15)
  (check-= (ref a 'gap) 1/15 1e-12)
  (check-true (ref a 'within))
  (define off (attribution records "add-8x8"))
  (check-near (ref off 'sum) 4)
  (check-near (ref off 'total) 12)
  (check-true (> (ref off 'gap) gap-tolerance))
  (check-false (ref off 'within)))

(test-case "the markdown flags a gap over the tolerance and says the contract is outside the sum"
  (define md (op-cost-markdown records))
  (check-true (string-contains? md "| **gap, over 10%** | 66.7% |"))
  (check-true (string-contains? md "| gap | 6.7% |"))
  (check-true (string-contains? md "contract (contracted add − dispatch), not in the sum | 5.00"))
  (check-true (string-contains? md "1-min load 20.0–30.0; busy cores 12.5–12.5")))

(test-case "K1 is the scalar add's facade less libtorch's own time, the raw call less two crossings"
  (check-near (crossing-us records) 0.1)
  (define one (k1 records))
  (check-near (ref one 'facade) 15)
  (check-= (ref one 'libtorch) 0.8 1e-12)
  (check-= (ref one 'value) 14.2 1e-12))

(test-case "K2 divides rktorch's one-thread facade by PyTorch's"
  (check-near (map (lambda (k) (ref k 'value)) (k2 records)) '(5 3)))

(test-case "K3 divides the per-op cost at the most threads by one thread's, per layer"
  (define facade (for/first ([k (in-list (k3 records))] #:when (equal? (ref k 'op) "scalar-add")) k))
  (check-equal? (ref facade 'threads) 8)
  (check-near (ref facade 'value) 2)
  (define raw (for/first ([k (in-list (k3 records #:layer "raw"))]
                          #:when (equal? (ref k 'op) "scalar-add"))
                k))
  (check-near (ref raw 'value) 1.5))

(test-case "K4 is the facade's heap bytes per op, beside the tensors it registers"
  (define four (for/first ([k (in-list (k4 records))] #:when (equal? (ref k 'op) "scalar-add")) k))
  (check-near (ref four 'value) 8000)
  (check-equal? (ref four 'tensors) 1)
  (check-near (layer-median records "add-8x8" "facade" #:key 'heap_bytes_per_op) 4000))

(test-case "K5 divides pipeline throughput by PyTorch's per worker count, skipping failed runs"
  (check-equal? (map (lambda (k) (ref k 'workers)) (k5 records)) '(0 8))
  (check-near (map (lambda (k) (ref k 'value)) (k5 records)) '(0.25 1)))

(test-case "K6 is the median GC share of wall time at the most workers"
  (define six (k6 records))
  (check-equal? (ref six 'workers) 8)
  (check-= (ref six 'value) 0.2 1e-12))

(test-case "the markdown lists a failed pipeline run"
  (check-true (string-contains? (op-cost-markdown records)
                                "- cifar w=8 rktorch rep 3: op-cost-cell exited with 1")))
