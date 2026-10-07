#lang racket/base

;; Run: raco test torch/tests/attention-parity-test.rkt (inside `nix develop`
;; or `.#cuda`, which provide Python torch; SKIPS when python3 can't import
;; torch). Every draw MUST stay in the order of the twins,
;; torch/tests/python/scaled_dot_product_attention.py,
;; torch/tests/python/multihead_attention.py and
;; torch/tests/python/transformer_layers.py.

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
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

  (define (padded-keys batch length padded)
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

  (define (run-mha device embed heads length source
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
    (define (apply-layer)
      (mha q k v
           #:key-padding-mask key-padding-mask
           #:attn-mask attn-mask
           #:causal? causal?
           #:need-weights? need-weights?
           #:average-attn-weights? average?))
    (define-values (out weights)
      (if need-weights? (apply-layer) (values (apply-layer) #f)))
    (define (weighed t)
      (sum (* t (to-device (randn (tensor-shape t)) device))))
    (backward! (if weights (+ (weighed out) (weighed weights)) (weighed out)))
    (hash 'params params
          'out out
          'weights weights
          'input-grads (map grad leaves)
          'param-grads (named-values mha grad)))

  (define mha-cases
    (list
     (list 'self_padding
           (lambda (device)
             (run-mha device 8 2 5 5 #:batch 3
                      #:padding (lambda () (padded-keys 3 5 2))
                      #:need-weights? #t)))
     (list 'self_padding_fused
           (lambda (device)
             (run-mha device 8 2 5 5 #:batch 3
                      #:padding (lambda () (padded-keys 3 5 2)))))
     (list 'causal
           (lambda (device)
             (run-mha device 8 4 5 5 #:batch-first? #t #:causal? #t)))
     (list 'causal_weights
           (lambda (device)
             (run-mha device 8 4 5 5 #:batch-first? #t #:causal? #t
                      #:need-weights? #t #:average? #f)))
     (list 'causal_padding
           (lambda (device)
             (run-mha device 8 4 5 5 #:batch-first? #t #:causal? #t
                      #:padding (lambda () (padded-keys 2 5 1)))))
     (list 'cross
           (lambda (device)
             (run-mha device 8 2 3 4 #:key-dim 5 #:value-dim 6 #:cross? #t
                      #:mask (lambda () (randn 4 3 4))
                      #:need-weights? #t #:average? #f)))
     (list 'cross_batch_first
           (lambda (device)
             (run-mha device 8 2 3 4 #:batch-first? #t #:key-dim 5 #:value-dim 6
                      #:cross? #t
                      #:padding (lambda () (randn 2 4))
                      #:mask (lambda ()
                               (eq (tril (ones 3 4 #:dtype 'bool)) 0)))))
     (list 'cross_same_width
           (lambda (device)
             (run-mha device 8 2 3 4 #:cross? #t #:need-weights? #t)))
     (list 'unbatched
           (lambda (device)
             (run-mha device 8 2 4 4 #:unbatched? #t
                      #:padding (lambda () (eq (tensor '(0 0 0 1)) 1))
                      #:need-weights? #t)))
     (list 'no_bias
           (lambda (device)
             (run-mha device 8 2 3 3 #:bias? #f #:need-weights? #t)))
     (list 'dropout_fused
           (lambda (device) (run-mha device 8 2 4 4 #:dropout 0.5)))
     (list 'dropout_weights
           (lambda (device)
             (run-mha device 8 2 4 4 #:dropout 0.5 #:need-weights? #t)))))

  (define (check-numbers label actual expected tolerance)
    (check-equal? (length actual) (length expected)
                  (format "~a: value count" label))
    (for ([a (in-list actual)] [b (in-list expected)] [i (in-naturals)])
      (check-= a b tolerance (format "~a ~a" label i))))

  (define (check-named label actual expected tolerance)
    (check-equal? (sort (map car actual) string<?)
                  (sort (map symbol->string (hash-keys expected)) string<?)
                  (format "~a: names" label))
    (for ([entry (in-list actual)])
      (check-numbers (format "~a ~a" label (car entry))
                     (cdr entry)
                     (hash-ref expected (string->symbol (car entry)) '())
                     tolerance)))

  (define (check-mha-twin device tolerance)
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
       (lambda () (python-check "multihead_attention.py"))))
    (with-default-device 'cpu
      (for ([row (in-list mha-cases)])
        (match-define (list key run-case) row)
        (define expected (hash-ref j key))
        (define label (format "mha ~a [~a]" key device))
        (define got (run-case device))
        (check-named (format "~a: init" label) (hash-ref got 'params)
                     (hash-ref expected 'params) tolerance)
        (check-equal? (tensor-shape (hash-ref got 'out))
                      (hash-ref expected 'out_shape) label)
        (check-numbers (format "~a: output" label) (flat (hash-ref got 'out))
                       (hash-ref expected 'out) tolerance)
        (define weights (hash-ref got 'weights))
        (cond
          [weights
           (check-equal? (tensor-shape weights)
                         (hash-ref expected 'weights_shape) label)
           (check-numbers (format "~a: weights" label) (flat weights)
                          (hash-ref expected 'weights) tolerance)]
          [else
           (check-equal? (hash-ref expected 'weights) 'null label)])
        (for ([g (in-list (hash-ref got 'input-grads))]
              [e (in-list (hash-ref expected 'input_grads))]
              [i (in-naturals)])
          (check-numbers (format "~a: input ~a gradient" label i) (flat g) e
                         tolerance))
        (check-named (format "~a: gradient" label)
                     (hash-ref got 'param-grads)
                     (hash-ref expected 'param_grads) tolerance))))

  (define (sequence-shape batch-first? batch length width)
    (if batch-first? (list batch length width) (list length batch width)))

  (define (later length source)
    (triu (ones length source #:dtype 'bool) 1))

  (define (built constructor stack form width layers norm?
                 #:heads [heads 2] #:dropout p #:activation activation
                 #:norm-first? norm-first? #:batch-first? batch-first?
                 #:bias? bias? #:eps eps)
    (define (make-layer)
      (constructor width #:heads heads #:ffn-width 16 #:dropout p
                   #:activation activation #:norm-first? norm-first?
                   #:batch-first? batch-first? #:bias? bias?
                   #:layer-norm-eps eps))
    (define (final-norm)
      (and norm? (LayerNorm width #:eps eps #:bias? bias?)))
    (case (and layers form)
      [(width)
       (stack width #:heads heads #:ffn-width 16 #:dropout p
              #:activation activation #:norm-first? norm-first?
              #:batch-first? batch-first? #:bias? bias?
              #:layer-norm-eps eps #:layers layers #:norm norm?)]
      [(layer) (stack (make-layer) #:layers layers #:norm (final-norm))]
      [(procedure) (stack make-layer #:layers layers #:norm (final-norm))]
      [else (make-layer)]))

  (define (backward-from m device out leaves)
    (backward! (sum (* out (to-device (randn (tensor-shape out)) device))))
    (hash 'out out
          'input-grads (map grad leaves)
          'param-grads (named-values m grad)))

  (define (run-encoder device
                       #:heads [heads 2]
                       #:batch [batch 3]
                       #:layers [layers #f]
                       #:norm? [norm? #f]
                       #:unbatched? [unbatched? #f]
                       #:padding [make-padding void]
                       #:mask [make-mask void]
                       #:causal? [causal? #f]
                       #:train? [train? #t]
                       #:dropout [p 0.0]
                       #:activation [activation 'relu]
                       #:norm-first? [norm-first? #f]
                       #:batch-first? [batch-first? #f]
                       #:bias? [bias? #t]
                       #:eps [eps 1e-5]
                       #:form [form 'width])
    (manual-seed! 0)
    (define m
      (built TransformerEncoderLayer TransformerEncoder form 8 layers norm?
             #:heads heads #:dropout p #:activation activation
             #:norm-first? norm-first? #:batch-first? batch-first?
             #:bias? bias? #:eps eps))
    (define params (named-values m values))
    (to m device)
    (unless train? (eval! m))
    (define src
      (requires-grad!
       (to-device (randn (if unbatched?
                             (list 5 8)
                             (sequence-shape batch-first? batch 5 8)))
                  device)))
    (define (on-device t) (and (tensor? t) (to-device t device)))
    (define padding (on-device (make-padding)))
    (define mask (on-device (make-mask)))
    (define out
      (m src #:mask mask #:key-padding-mask padding #:causal? causal?))
    (hash-set (backward-from m device out (list src)) 'params params))

  (define (run-decoder device
                       #:layers [layers #f]
                       #:norm? [norm? #f]
                       #:tgt-padding [make-tgt-padding void]
                       #:memory-padding [make-memory-padding void]
                       #:tgt-mask [make-tgt-mask void]
                       #:memory-mask [make-memory-mask void]
                       #:tgt-causal? [tgt-causal? #f]
                       #:memory-causal? [memory-causal? #f]
                       #:train? [train? #t]
                       #:dropout [p 0.0]
                       #:activation [activation 'relu]
                       #:norm-first? [norm-first? #f]
                       #:batch-first? [batch-first? #f]
                       #:bias? [bias? #t]
                       #:eps [eps 1e-5]
                       #:form [form 'width])
    (manual-seed! 0)
    (define m
      (built TransformerDecoderLayer TransformerDecoder form 8 layers norm?
             #:dropout p #:activation activation #:norm-first? norm-first?
             #:batch-first? batch-first? #:bias? bias? #:eps eps))
    (define params (named-values m values))
    (to m device)
    (unless train? (eval! m))
    (define (leaf n)
      (requires-grad!
       (to-device (randn (sequence-shape batch-first? 2 n 8)) device)))
    (define tgt (leaf 4))
    (define memory (leaf 5))
    (define (on-device t) (and (tensor? t) (to-device t device)))
    (define tgt-padding (on-device (make-tgt-padding)))
    (define memory-padding (on-device (make-memory-padding)))
    (define tgt-mask (on-device (make-tgt-mask)))
    (define memory-mask (on-device (make-memory-mask)))
    (define out
      (m tgt memory
         #:tgt-mask tgt-mask
         #:memory-mask memory-mask
         #:tgt-key-padding-mask tgt-padding
         #:memory-key-padding-mask memory-padding
         #:tgt-causal? tgt-causal?
         #:memory-causal? memory-causal?))
    (hash-set (backward-from m device out (list tgt memory)) 'params params))

  (define (fast-path device)
    (manual-seed! 0)
    (define m (TransformerEncoderLayer 8 #:heads 2 #:ffn-width 16
                                       #:dropout 0.1 #:batch-first? #t))
    (define params (named-values m values))
    (to m device)
    (eval! m)
    (define src (to-device (randn 3 5 8) device))
    (define padding (to-device (padded-keys 3 5 2) device))
    (hash 'params params 'out (m src #:key-padding-mask padding)))

  (define transformer-cases
    (list
     (list 'encoder_post_relu
           (lambda (d)
             (run-encoder d #:padding (lambda () (padded-keys 3 5 2)))))
     (list 'encoder_pre_gelu_causal
           (lambda (d)
             (run-encoder d #:norm-first? #t #:activation 'gelu
                          #:batch-first? #t #:batch 2 #:causal? #t)))
     (list 'encoder_pre_gelu_tanh
           (lambda (d)
             (run-encoder d #:norm-first? #t #:activation 'gelu-tanh
                          #:batch-first? #t #:batch 2
                          #:mask (lambda () (randn 5 5))
                          #:padding (lambda () (padded-keys 2 5 1)))))
     (list 'encoder_causal_padding
           (lambda (d)
             (run-encoder d #:heads 4 #:causal? #t
                          #:padding (lambda () (padded-keys 3 5 2)))))
     (list 'encoder_eval
           (lambda (d)
             (run-encoder d #:dropout 0.1 #:train? #f #:causal? #t
                          #:padding (lambda () (padded-keys 3 5 1)))))
     (list 'encoder_no_bias_unbatched
           (lambda (d)
             (run-encoder d #:bias? #f #:unbatched? #t #:eps 1e-6)))
     (list 'encoder_dropout
           (lambda (d) (run-encoder d #:dropout 0.5 #:batch 2)))
     (list 'decoder_post_relu
           (lambda (d)
             (run-decoder d #:tgt-causal? #t
                          #:memory-padding (lambda () (padded-keys 2 5 2)))))
     (list 'decoder_pre_gelu
           (lambda (d)
             (run-decoder d #:norm-first? #t #:activation 'gelu
                          #:batch-first? #t
                          #:tgt-mask (lambda () (randn 4 4))
                          #:tgt-padding (lambda () (padded-keys 2 4 1))
                          #:memory-mask (lambda () (later 4 5)))))
     (list 'decoder_memory_causal
           (lambda (d)
             (run-decoder d #:memory-causal? #t #:tgt-causal? #t)))
     (list 'decoder_eval
           (lambda (d)
             (run-decoder d #:dropout 0.1 #:train? #f #:tgt-causal? #t
                          #:memory-padding (lambda () (padded-keys 2 5 1)))))
     (list 'decoder_dropout
           (lambda (d) (run-decoder d #:dropout 0.5 #:norm-first? #t)))
     (list 'encoder_stack
           (lambda (d)
             (run-encoder d #:layers 3 #:norm? #t
                          #:padding (lambda () (padded-keys 3 5 2)))))
     (list 'encoder_stack_pre_causal
           (lambda (d)
             (run-encoder d #:layers 2 #:norm-first? #t #:batch-first? #t
                          #:batch 2 #:activation 'gelu-tanh #:causal? #t)))
     (list 'decoder_stack
           (lambda (d)
             (run-decoder d #:layers 2 #:norm? #t #:tgt-causal? #t
                          #:memory-padding (lambda () (padded-keys 2 5 2)))))
     (list 'decoder_stack_pre
           (lambda (d)
             (run-decoder d #:layers 3 #:norm-first? #t #:norm? #t
                          #:batch-first? #t #:activation 'gelu
                          #:tgt-causal? #t #:bias? #f #:eps 1e-6)))
     (list 'layer_encoder
           (lambda (d)
             (run-encoder d #:layers 3 #:norm? #t #:form 'layer
                          #:norm-first? #t #:bias? #f
                          #:padding (lambda () (padded-keys 3 5 2)))))
     (list 'layer_decoder
           (lambda (d)
             (run-decoder d #:layers 2 #:form 'layer #:batch-first? #t
                          #:tgt-causal? #t
                          #:memory-padding (lambda () (padded-keys 2 5 2)))))
     (list 'procedure_encoder
           (lambda (d)
             (run-encoder d #:layers 2 #:norm? #t #:form 'procedure
                          #:padding (lambda () (padded-keys 3 5 2)))))
     (list 'procedure_decoder
           (lambda (d)
             (run-decoder d #:layers 2 #:form 'procedure #:norm-first? #t
                          #:activation 'gelu-tanh #:batch-first? #t
                          #:tgt-causal? #t
                          #:memory-padding (lambda () (padded-keys 2 5 1)))))
     (list 'fast_path fast-path)))

  (define (check-transformer-twin device tolerance)
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
       (lambda () (python-check "transformer_layers.py"))))
    (with-default-device 'cpu
      (for ([row (in-list transformer-cases)])
        (match-define (list key run-case) row)
        (define expected (hash-ref j key))
        (define label (format "transformer ~a [~a]" key device))
        (define got (run-case device))
        (check-named (format "~a: init" label) (hash-ref got 'params)
                     (hash-ref expected 'params) 0.0)
        (check-equal? (tensor-shape (hash-ref got 'out))
                      (hash-ref expected 'out_shape) label)
        (check-numbers (format "~a: output" label) (flat (hash-ref got 'out))
                       (hash-ref expected 'out) tolerance)
        (when (hash-has-key? got 'input-grads)
          (for ([g (in-list (hash-ref got 'input-grads))]
                [e (in-list (hash-ref expected 'input_grads))]
                [i (in-naturals)])
            (check-numbers (format "~a: input ~a gradient" label i) (flat g)
                           e tolerance))
          (check-named (format "~a: gradient" label)
                       (hash-ref got 'param-grads)
                       (hash-ref expected 'param_grads) tolerance)))))

  (cond
    [(not (python-torch-available?))
     (printf "[attention-parity-test] skipped: python3 `torch` ~a\n"
             "not available (run inside `nix develop`)")]
    [else
     (check-attention-twin 'cpu tol)
     (check-mha-twin 'cpu tol)
     (check-transformer-twin 'cpu tol)
     (when (and (cuda-available?)
                (python-cuda-available?))
       (check-attention-twin 'cuda tol)
       (check-mha-twin 'cuda tol)
       (check-transformer-twin 'cuda tol))]))
