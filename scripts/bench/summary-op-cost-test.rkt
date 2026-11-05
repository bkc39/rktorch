#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-contains?)
         rackunit
         (only-in "markdown.rkt" ref)
         (only-in "summary-op-cost.rkt"
                  attribution crossing-us gap-tolerance k1-overhead k2-pytorch-ratio
                  k3-thread-scaling k4-heap k5-pipelines k6-gc-share layer-median op-cost-markdown
                  paired-delta piece-label samples-by-rep)
         (only-in "summary.rkt" read-records))

(define-runtime-path fixture "fixtures/op-cost/records.jsonl")

(define records (read-records fixture))

(define-check (check-near actual expected)
  (unless (if (list? expected)
              (and (list? actual) (= (length actual) (length expected))
                   (andmap (lambda (a e) (< (abs (- a e)) 1e-9)) actual expected))
              (and (real? actual) (< (abs (- actual expected)) 1e-9)))
    (fail-check)))

(define (row-of rows key value)
  (findf (lambda (r) (equal? (ref r key) value)) rows))

(define (piece-of a layer)
  (row-of (ref a 'pieces) 'layer layer))

(define tolerance-text (format "~a%" (exact->inexact (* 100 gap-tolerance))))

(test-case "the records file parses every line, and a blank line is no record"
  (check-equal? (length records) 76))

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
                "oom-retry wrapper + accounting (account!)"))

(test-case "the pieces of a ladder are the paired steps between the layers present"
  (define a (attribution records "scalar-add"))
  (check-equal? (map (lambda (p) (ref p 'layer)) (ref a 'pieces))
                '("raw" "registered" "accounted" "checked" "wrapped" "unchecked" "facade"))
  (check-near (map (lambda (p) (ref p 'us)) (ref a 'pieces)) '(1 2 3 4 2 1 1))
  (check-equal? (ref (piece-of a "accounted") 'piece)
                "oom-retry wrapper + accounting (account!)")
  (check-near (ref (piece-of a "registered") 'heap) 1000)
  (check-equal? (ref (piece-of a "raw") 'crossings) 2)
  (check-equal? (ref (piece-of a "wrapped") 'crossings) 2)
  (check-near (ref a 'contract) 5))

(test-case "the paired steps telescope to the top layer, so their sum is only median noise"
  (define a (attribution records "scalar-add"))
  (check-near (ref a 'sum) 14)
  (check-near (ref a 'total) 15)
  (check-= (ref a 'noise) -1/15 1e-12)
  (check-near (ref (attribution records "add-8x8") 'noise) -2/3))

(test-case "leave-one-out flags a ladder piece the whole stack does not bear out"
  (define pressure (row-of (ref (attribution records "scalar-add") 'loo) 'layer "checked"))
  (check-near (ref pressure 'ladder) 4)
  (check-near (ref pressure 'loo) 0.5)
  (check-false (ref pressure 'agrees))
  (define retry (row-of (ref (attribution records "zeros-8x8") 'loo) 'layer "retry"))
  (check-near (ref retry 'ladder) 1)
  (check-near (ref retry 'loo) 1)
  (check-true (ref retry 'agrees)))

(test-case "a piece spanning several layers has no leave-one-out check"
  (define a (attribution records "scalar-add"))
  (check-equal? (map (lambda (c) (ref c 'layer)) (ref a 'loo)) '("checked"))
  (check-equal? (ref (piece-of (attribution records "zeros-8x8") "wrapped") 'piece)
                (piece-label '(accounted checked wrapped))))

(test-case "the markdown labels the noise, flags a disagreement, and sets the contract beside"
  (define md (op-cost-markdown records))
  (check-true (string-contains? md "| median noise (paired steps − facade's median) | -6.7% |"))
  (check-true (string-contains? md (format "| pressure check | 4.00 | 0.50 | **no, over ~a** |"
                                           tolerance-text)))
  (check-true (string-contains? md "| oom-retry wrapper | 1.00 | 1.00 | yes |"))
  (check-true (string-contains? md "contract (contracted add − dispatch), beside the ladder | 5.00"))
  (check-true (string-contains? md "1-min load 20.0–30.0; busy cores 12.5–12.5")))

(test-case "K1 is the scalar add's facade less libtorch's own time, the raw call less two crossings"
  (check-near (crossing-us records) 0.1)
  (define one (k1-overhead records))
  (check-near (ref one 'facade) 15)
  (check-= (ref one 'libtorch) 0.8 1e-12)
  (check-= (ref one 'value) 14.2 1e-12))

(test-case "K2 divides rktorch's one-thread facade by PyTorch's"
  (check-near (map (lambda (k) (ref k 'value)) (k2-pytorch-ratio records)) '(5 3)))

(test-case "K3 divides the per-op cost at the most threads by one thread's, per layer"
  (define facade (row-of (k3-thread-scaling records) 'op "scalar-add"))
  (check-equal? (ref facade 'threads) 8)
  (check-near (ref facade 'value) 2)
  (define raw (row-of (k3-thread-scaling records #:layer "raw") 'op "scalar-add"))
  (check-near (ref raw 'value) 1.5))

(test-case "K3's finalized share counts finalizer runs per tensor the op registers"
  (define (row threads finalized)
    (hasheq 'kind "op" 'impl "rktorch" 'op "chain" 'layer "facade" 'threads threads 'rep 0
            'us_per_op 10.0 'finalized_per_op finalized))
  (define extra (list (row 1 44.0) (row 8 11.0)
                      (hasheq 'kind "counts" 'op "chain" 'layer "facade" 'tensors_per_op 44)))
  (check-near (ref (row-of (k3-thread-scaling (append records extra)) 'op "chain") 'finalized)
              0.25))

(test-case "K4 is the facade's heap bytes per tensor it registers"
  (define four (row-of (k4-heap records) 'op "scalar-add"))
  (check-near (ref four 'value) 8000)
  (check-equal? (ref four 'tensors) 1)
  (define chain (row-of (k4-heap (append records (list (hasheq 'kind "counts" 'op "add-8x8"
                                                              'layer "facade"
                                                              'tensors_per_op 4))))
                        'op "add-8x8"))
  (check-near (ref chain 'bytes) 4000)
  (check-near (ref chain 'value) 1000)
  (check-near (layer-median records "add-8x8" "facade" #:key 'heap_bytes_per_op) 4000))

(test-case "K5 divides pipeline throughput by PyTorch's per worker count, skipping failed runs"
  (check-equal? (map (lambda (k) (ref k 'workers)) (k5-pipelines records)) '(0 8))
  (check-near (map (lambda (k) (ref k 'value)) (k5-pipelines records)) '(0.25 1))
  (check-true (string-contains? (op-cost-markdown records) "rktorch converts first (#458)")))

(test-case "K6 is the median GC share of wall time at the most workers"
  (define six (k6-gc-share records))
  (check-equal? (ref six 'workers) 8)
  (check-= (ref six 'value) 0.2 1e-12))

(test-case "the markdown lists a failed pipeline run"
  (check-true (string-contains? (op-cost-markdown records)
                                "- cifar w=8 rktorch rep 3: op-cost-cell exited with 1")))
