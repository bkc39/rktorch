#lang racket/base

(module+ test
  (require (only-in racket/contract/base flat-contract-predicate)
           (only-in rackunit check-equal? check-exn check-not-false)
           (only-in "../foreign/ops.rkt" float-dtype-max)
           (only-in "../main.rkt" fill-value/c full item)
           "private/python-env.rkt")

  (define documented-deviations
    (hasheq 'float16 '("rounds")
            'bfloat16 '("rounds")
            'float32 '()
            'float64 '()
            'int64 '("truncates")
            'uint8 '("truncates" "wraps")
            'bool '("any-real-bool")))

  (define (probe-row dtype row)
    (define text (hash-ref row 'value))
    (define v (string->number text))
    (define fills (hash-ref row 'fills #f))
    (define deviation (and (string? (hash-ref row 'deviation #f))
                           (hash-ref row 'deviation)))
    (define label (format "(full ~a 1 #:dtype '~a)" text dtype))
    (define ours? ((flat-contract-predicate (fill-value/c dtype)) v))
    (check-equal? ours? (and fills (not deviation) #t)
                  (format "~a: accepted exactly where torch.full fills as asked"
                          label))
    (cond
      [ours?
       (check-equal? (item (full v 1 #:dtype dtype)) (string->number fills)
                     (format "~a fills what torch.full fills" label))]
      [else
       (check-exn #rx"^full: contract violation"
                  (lambda () (full v 1 #:dtype dtype))
                  (format "~a blames the caller" label))])
    (check-not-false (or (not deviation)
                         (member deviation
                                 (hash-ref documented-deviations dtype)))
                     (format "~a: torch.full ~a, undocumented at ~a"
                             label deviation dtype)))

  (cond
    [(not (python-torch-available?))
     (printf "[fill-value-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (define j (python-check "fill_value_ranges.py"))
     (check-equal? float-dtype-max
                   (for/hasheq ([(dtype m) (in-hash (hash-ref j 'max))])
                     (values dtype (string->number m)))
                   "each float dtype's limit is torch.finfo(dtype).max")
     (for* ([(dtype rows) (in-hash (hash-ref j 'probes))]
            [row (in-list rows)])
       (probe-row dtype row))]))
