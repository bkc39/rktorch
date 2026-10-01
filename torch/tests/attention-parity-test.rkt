#lang racket/base

;; Run: raco test torch/tests/attention-parity-test.rkt (inside `nix develop`
;; or `.#cuda`, which provide Python torch; SKIPS when python3 can't import
;; torch). Every draw MUST stay in the order of
;; torch/tests/python/scaled_dot_product_attention.py.

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "private/python-env.rkt")

  (define (no-mask) #f)

  (define (padding)
    (reshape (eq (tensor '((1 1 1 1 1) (1 1 1 0 0))) 1) 2 1 1 5))

  (define cases
    (list (list 'plain 3 no-mask #f #f)
          (list 'bool_mask 3 padding #f #f)
          (list 'float_mask 3 (lambda () (randn 3 5)) #f #f)
          (list 'causal 5 no-mask #t #f)
          (list 'scale 3 no-mask #f 0.3)))

  (define (attend device query-length make-mask causal? scale)
    (define (leaf . shape)
      (requires-grad! (to-device (randn shape) device)))
    (define q (leaf 2 2 query-length 4))
    (define k (leaf 2 2 5 4))
    (define v (leaf 2 2 5 6))
    (define mask (make-mask))
    (define out
      (scaled-dot-product-attention q k v
                                    #:mask (and mask (to-device mask device))
                                    #:causal? causal?
                                    #:scale scale))
    (define weights (to-device (randn (tensor-shape out)) device))
    (backward! (sum (* out weights)))
    (values out (list q k v)))

  (define (check-values label t expected tolerance)
    (for ([a (in-list (tensor->list (to-device t 'cpu)))]
          [b (in-list expected)]
          [i (in-naturals)])
      (check-= a b tolerance (format "~a ~a" label i))))

  (define (check-attention-twin device tolerance)
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
       (lambda () (python-check "scaled_dot_product_attention.py"))))
    (with-default-device 'cpu
      (manual-seed! 0)
      (for ([row (in-list cases)])
        (match-define (list key query-length make-mask causal? scale) row)
        (define expected (hash-ref j key))
        (define label (format "sdpa ~a [~a]" key device))
        (define-values (out leaves)
          (attend device query-length make-mask causal? scale))
        (check-equal? (tensor-shape out) (hash-ref expected 'shape) label)
        (check-values (format "~a: output" label) out (hash-ref expected 'out)
                      tolerance)
        (for ([leaf (in-list leaves)]
              [g (in-list (hash-ref expected 'grads))]
              [name (in-list '(query key value))])
          (check-values (format "~a: ~a gradient" label name) (grad leaf) g
                        tolerance)))))

  (cond
    [(not (python-torch-available?))
     (printf "[attention-parity-test] skipped: python3 `torch` ~a\n"
             "not available (run inside `nix develop`)")]
    [else
     (check-attention-twin 'cpu tol)
     (when (and (cuda-available?)
                (python-cuda-available?))
       (check-attention-twin 'cuda tol))]))
