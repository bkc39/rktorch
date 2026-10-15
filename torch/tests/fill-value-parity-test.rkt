#lang racket/base

(module+ test
  (require (only-in racket/contract/base flat-contract-predicate)
           (only-in racket/match match-define)
           (only-in racket/math infinite? nan?)
           (only-in rackunit check-equal? check-exn check-true)
           (only-in "../main.rkt" fill-value/c full item)
           "private/python-env.rkt")

  (define float-max
    (hasheq 'float16 65504.0
            'bfloat16 3.3895313892515355e38
            'float32 3.4028234663852886e38
            'float64 1.7976931348623157e308))

  (define (read-value s) (read (open-input-string s)))

  (define (documented-stricter? dtype v)
    (case dtype
      [(float16 bfloat16)
       (not (or (nan? v) (infinite? v)
                (<= (abs v) (hash-ref float-max dtype))))]
      [(float32 float64) #f]
      [(uint8) (or (not (integer? v)) (negative? v))]
      [(bool) (not (and (integer? v) (<= 0 v 1)))]
      [else (not (integer? v))]))

  (cond
    [(not (python-torch-available?))
     (printf "[fill-value-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (for* ([(dtype probes) (in-hash (python-check "fill_value_ranges.py"))]
            [probe (in-list probes)])
       (match-define (list text theirs) probe)
       (define v (read-value text))
       (define label (format "(full ~a 1 #:dtype '~a)" text dtype))
       (define ours? ((flat-contract-predicate (fill-value/c dtype)) v))
       (cond
         [ours?
          (check-equal? (item (full v 1 #:dtype dtype)) (read-value theirs)
                        (format "~a fills what torch.full fills" label))]
         [else
          (check-exn #rx"^full: contract violation"
                     (lambda () (full v 1 #:dtype dtype))
                     (format "~a blames the caller" label))
          (check-true (or (equal? theirs "refused")
                          (documented-stricter? dtype v))
                      (format "~a refused where torch.full gives ~a"
                              label theirs))]))]))
