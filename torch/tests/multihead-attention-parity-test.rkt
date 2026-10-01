#lang racket/base

;; Run: raco test torch/tests/multihead-attention-parity-test.rkt (inside
;; `nix develop` or `.#cuda`, which provide Python torch; SKIPS when python3
;; can't import torch). Every draw MUST stay in the order of
;; torch/tests/python/multihead_attention.py.

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           "private/python-env.rkt")

  (define (padding batch length padded)
    (eq (tensor (for/list ([b (in-range batch)])
                  (for/list ([i (in-range length)])
                    (if (and (= b (sub1 batch)) (>= i (- length padded)))
                        1
                        0))))
        1))

  (define (flat t)
    (for/list ([x (in-flattened-tensor (to-device t 'cpu))]) x))

  (define (named-values mha value-of)
    (for/list ([(name p) (in-named-parameters mha)])
      (cons name (flat (value-of p)))))

  (define (run device embed heads length source
               #:batch [batch 2]
               #:batch-first? [batch-first? #f]
               #:key-dim [key-dim embed]
               #:value-dim [value-dim embed]
               #:bias? [bias? #t]
               #:dropout [dropout 0.0]
               #:cross? [cross? #f]
               #:unbatched? [unbatched? #f]
               #:padding [make-padding void]
               #:mask [make-mask void]
               #:causal? [causal? #f]
               #:need-weights? [need-weights? #f]
               #:average? [average? #t])
    (manual-seed! 0)
    (define mha
      (MultiheadAttention embed #:heads heads #:dropout dropout #:bias? bias?
                          #:batch-first? batch-first?
                          #:key-dim key-dim #:value-dim value-dim))
    (define params (named-values mha values))
    (to mha device)
    (define (shape n width)
      (cond
        [unbatched? (list n width)]
        [batch-first? (list batch n width)]
        [else (list n batch width)]))
    (define (leaf n width)
      (requires-grad! (to-device (randn (shape n width)) device)))
    (define leaves
      (if cross?
          (list (leaf length embed) (leaf source key-dim)
                (leaf source value-dim))
          (list (leaf length embed))))
    (match-define (list q k v)
      (if cross? leaves (list (car leaves) (car leaves) (car leaves))))
    (define (on-device t) (and (tensor? t) (to-device t device)))
    (define key-padding-mask (on-device (make-padding)))
    (define attn-mask (on-device (make-mask)))
    (define (attend)
      (mha q k v
           #:key-padding-mask key-padding-mask
           #:attn-mask attn-mask
           #:causal? causal?
           #:need-weights? need-weights?
           #:average-attn-weights? average?))
    (define-values (out weights)
      (if need-weights? (attend) (values (attend) #f)))
    (define (weighed t)
      (sum (* t (to-device (randn (tensor-shape t)) device))))
    (backward! (if weights (+ (weighed out) (weighed weights)) (weighed out)))
    (hash 'params params
          'out out
          'weights weights
          'input-grads (map grad leaves)
          'param-grads (named-values mha grad)))

  (define cases
    (list
     (list 'self_padding
           (lambda (device)
             (run device 8 2 5 5 #:batch 3
                  #:padding (lambda () (padding 3 5 2))
                  #:need-weights? #t)))
     (list 'self_padding_fused
           (lambda (device)
             (run device 8 2 5 5 #:batch 3
                  #:padding (lambda () (padding 3 5 2)))))
     (list 'causal
           (lambda (device)
             (run device 8 4 5 5 #:batch-first? #t #:causal? #t)))
     (list 'causal_weights
           (lambda (device)
             (run device 8 4 5 5 #:batch-first? #t #:causal? #t
                  #:need-weights? #t #:average? #f)))
     (list 'causal_padding
           (lambda (device)
             (run device 8 4 5 5 #:batch-first? #t #:causal? #t
                  #:padding (lambda () (padding 2 5 1)))))
     (list 'cross
           (lambda (device)
             (run device 8 2 3 4 #:key-dim 5 #:value-dim 6 #:cross? #t
                  #:mask (lambda () (randn 4 3 4))
                  #:need-weights? #t #:average? #f)))
     (list 'cross_batch_first
           (lambda (device)
             (run device 8 2 3 4 #:batch-first? #t #:key-dim 5 #:value-dim 6
                  #:cross? #t
                  #:padding (lambda () (randn 2 4))
                  #:mask (lambda () (eq (tril (ones 3 4 #:dtype 'bool)) 0)))))
     (list 'cross_same_width
           (lambda (device)
             (run device 8 2 3 4 #:cross? #t #:need-weights? #t)))
     (list 'unbatched
           (lambda (device)
             (run device 8 2 4 4 #:unbatched? #t
                  #:padding (lambda () (eq (tensor '(0 0 0 1)) 1))
                  #:need-weights? #t)))
     (list 'no_bias
           (lambda (device)
             (run device 8 2 3 3 #:bias? #f #:need-weights? #t)))
     (list 'dropout_fused
           (lambda (device) (run device 8 2 4 4 #:dropout 0.5)))
     (list 'dropout_weights
           (lambda (device)
             (run device 8 2 4 4 #:dropout 0.5 #:need-weights? #t)))))

  (define (check-values label actual expected tolerance)
    (check-equal? (length actual) (length expected)
                  (format "~a: value count" label))
    (for ([a (in-list actual)] [b (in-list expected)] [i (in-naturals)])
      (check-= a b tolerance (format "~a ~a" label i))))

  (define (check-named label actual expected tolerance)
    (check-equal? (sort (map car actual) string<?)
                  (sort (map symbol->string (hash-keys expected)) string<?)
                  (format "~a: names" label))
    (for ([entry (in-list actual)])
      (check-values (format "~a ~a" label (car entry))
                    (cdr entry)
                    (hash-ref expected (string->symbol (car entry)) '())
                    tolerance)))

  (define (check-twin device tolerance)
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
       (lambda () (python-check "multihead_attention.py"))))
    (with-default-device 'cpu
      (for ([row (in-list cases)])
        (match-define (list key run-case) row)
        (define expected (hash-ref j key))
        (define label (format "mha ~a [~a]" key device))
        (define got (run-case device))
        (check-named (format "~a: init" label) (hash-ref got 'params)
                     (hash-ref expected 'params) tolerance)
        (check-equal? (tensor-shape (hash-ref got 'out))
                      (hash-ref expected 'out_shape) label)
        (check-values (format "~a: output" label) (flat (hash-ref got 'out))
                      (hash-ref expected 'out) tolerance)
        (define weights (hash-ref got 'weights))
        (cond
          [weights
           (check-equal? (tensor-shape weights)
                         (hash-ref expected 'weights_shape) label)
           (check-values (format "~a: weights" label) (flat weights)
                         (hash-ref expected 'weights) tolerance)]
          [else
           (check-equal? (hash-ref expected 'weights) 'null label)])
        (for ([g (in-list (hash-ref got 'input-grads))]
              [e (in-list (hash-ref expected 'input_grads))]
              [i (in-naturals)])
          (check-values (format "~a: input ~a gradient" label i) (flat g) e
                        tolerance))
        (check-named (format "~a: gradient" label)
                     (hash-ref got 'param-grads)
                     (hash-ref expected 'param_grads) tolerance))))

  (cond
    [(not (python-torch-available?))
     (printf "[multihead-attention-parity-test] skipped: python3 `torch` ~a\n"
             "not available (run inside `nix develop`)")]
    [else
     (check-twin 'cpu tol)
     (when (and (cuda-available?) (python-cuda-available?))
       (check-twin 'cuda tol))]))
