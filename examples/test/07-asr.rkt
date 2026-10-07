#lang racket/base

(require (only-in racket/list first last)
         (only-in racket/math nan?)
         (only-in racket/string string-prefix?)
         torch
         torch/nn
         (only-in torch/audio/librispeech
                  librispeech-utterances load-librispeech-fixture)
         (only-in torch/audio/metrics cer wer)
         "../racket/07-asr.rkt")

(module+ main
  (require (only-in "private/env.rkt" env-number))
  (printf "device: ~a\n" (pick-device))
  (define-values (net vocab)
    (train-librispeech #:epochs (env-number "EPOCHS" 20)
                       #:limit (env-number "LIMIT" #f)))
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define features (utterance-features samples rate))
  (define hypothesis (transcribe net vocab features))
  (printf "ref:     ~a\nctc:     ~a\nattend:  ~a\nwer: ~a  cer: ~a\n"
          transcript (greedy-decode net vocab features) hypothesis
          (wer transcript hypothesis) (cer transcript hypothesis)))

(module+ test
  (require rackunit)
  (define-values (losses net vocab device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  ;; conv1, conv2 and four dilated convolutions, each weight+bias; 6 encoder
  ;; layers x (4 attention projections, 2 feed-forward Linears and 2
  ;; LayerNorms, every one weight+bias) and the stack's norm; the CTC head;
  ;; the token table; 6 decoder layers x (8 projections, 2 Linears and 3
  ;; LayerNorms) and the stack's norm; the head: 273 tensors.
  (define names (map car (named-parameters net)))
  (check-equal? (length names) 273)
  (check-equal? (first names) "conv1.weight")
  (check-equal? (last names) "head.bias")
  (check-not-false (member "dilations.3.bias" names))
  (check-not-false (member "encoder.layers.5.self-attn.query.weight" names))
  (check-not-false (member "encoder.layers.0.linear1.weight" names))
  (check-not-false (member "encoder.norm.weight" names))
  (check-not-false (member "tok-emb.weight" names))
  (check-not-false
   (member "decoder.layers.5.multihead-attn.out.bias" names))
  (check-not-false (member "decoder.layers.0.self-attn.key.bias" names))
  (check-not-false (member "decoder.layers.0.norm3.weight" names))
  (check-not-false (member "decoder.norm.bias" names))
  ;; the hand-written blocks' names are gone
  (check-false (member "encoders.5.attention.wq.weight" names))
  (check-false (member "decoders.5.cross.wo.bias" names))
  (check-false (member "ln-enc.weight" names))
  (define v-size (vector-length vocab))
  (check-equal? (for/sum ([p (in-list (parameters net))]) (numel p))
                (+ (* 194 v-size) 778114))
  (check-equal? (tensor-shape (car (parameters net))) '(64 80 3))
  (check-equal? (tensor-shape
                 (cdr (assoc "tok-emb.weight" (named-parameters net))))
                (list (+ v-size 2) 64))
  ;; the standard stacks start every block as a copy of the first, as
  ;; nn.TransformerEncoder and nn.TransformerDecoder do
  (let ()
    (manual-seed! 0)
    (define fresh (named-parameters (asr 80 v-size)))
    (define (block-values stack layer)
      (define prefix (format "~a.layers.~a." stack layer))
      (for/list ([named (in-list fresh)]
                 #:when (string-prefix? (car named) prefix))
        (tensor->list (cdr named))))
    (check-equal? (length (block-values "encoder" 5)) 16)
    (check-equal? (length (block-values "decoder" 5)) 26)
    (for ([layer (in-range 1 6)])
      (check-equal? (block-values "encoder" layer) (block-values "encoder" 0))
      (check-equal? (block-values "decoder" layer)
                    (block-values "decoder" 0))))
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define features (utterance-features samples rate))
  (define ctc-hyp (greedy-decode net vocab features))
  (define att-hyp (transcribe net vocab features))
  (for ([hypothesis (in-list (list ctc-hyp att-hyp))])
    (check-true (string? hypothesis))
    (check-true (for/and ([c (in-string hypothesis)])
                  (and (member c (vector->list vocab)) #t))
                (format "decoded chars outside the vocab: ~v" hypothesis)))
  (check-true (layer-training? net) "decoding left the net in eval mode")
  (check-equal? (wer transcript transcript) 0)
  (check-equal? (cer transcript transcript) 0)
  (check-equal? (wer transcript "") 1)
  (check-equal? (cer transcript "") 1)
  (define mel (ref features 0))
  (define batch-loss
    (hybrid-batch-loss net vocab
                       (list mel (narrow mel 1 0 200))
                       (list transcript "MISTER QUILTER")))
  (check-true (rational? (item batch-loss)))
  (check-false (nan? (item batch-loss)))
  ;; the masking invariant the conv-stack fix restored: a row's encoder
  ;; output must not depend on how much padding its bucket neighbour forced
  (let ()
    (define short (narrow mel 1 0 200))
    (define t-max (cadr (tensor-shape mel)))
    (define sos-in
      (unsqueeze (to-dtype (tensor (list (add1 (vector-length vocab))))
                           'int64)
                 0))
    (define-values (alone _al) (net (unsqueeze short 0) sos-in #f))
    (define-values (batched _bl)
      (net (stack (list (cat (list short (zeros 80 (- t-max 200))) 1) mel) 0)
           (cat (list sos-in sos-in) 0)
           (list 200 t-max)))
    (define frames (cadr (tensor-shape alone)))
    (for ([a (in-list (tensor->list (ref alone 0)))]
          [b (in-list (tensor->list (narrow (ref batched 0) 0 0 frames)))]
          [i (in-naturals)])
      (check-= a b 1e-4
               (format "padding perturbed encoder output ~a" i))))
  ;; causal: changing the last decoder input leaves the earlier positions'
  ;; logits where they were
  (let ()
    (define (logits-of ids)
      (with-no-grad
        (define-values (_ctc logits)
          (net features (unsqueeze (to-dtype (tensor ids) 'int64) 0) #f))
        logits))
    (define sos (add1 v-size))
    (define (prefix t) (narrow t 1 0 3))
    (check-true (< (item (max (abs (- (prefix (logits-of (list sos 1 2 3)))
                                      (prefix (logits-of (list sos 1 2 4)))))))
                   1e-6))
    (check-true (> (item (max (abs (- (logits-of (list sos 1 2 3))
                                      (logits-of (list sos 1 2 4))))))
                   1e-4)))
  (check-true (<= (string-length att-hyp) 500)
              (format "transcribe failed to terminate: ~v" att-hyp))
  (check-true (device? (pick-device)))
  (let ()
    (manual-seed! 0)
    (define drop-net (asr 80 (vector-length vocab) #:dropout 0.5))
    (define sos-in
      (unsqueeze (to-dtype (tensor (list (add1 (vector-length vocab))))
                           'int64)
                 0))
    (define (logits)
      (define-values (_ctc l) (drop-net features sos-in #f))
      (tensor->list l))
    (check-not-equal? (logits) (logits)
                      "dropout did not perturb a training-mode forward")
    (in-eval-mode drop-net
      (check-equal? (logits) (logits)
                    "dropout stayed active in eval mode")))
  (check-exn #rx"no utterances to score"
             (lambda () (evaluate net vocab '())))
  ;; device RNG streams differ, so this arm checks convergence, never
  ;; equality
  (define accel (pick-device))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net a-vocab _a-dev)
      (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l))))
                        a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))
    (check-true (string? (greedy-decode a-net a-vocab features)))
    (check-true (string? (transcribe a-net a-vocab features)))
    ;; a multi-row batch must come back on the accelerator
    (define a-mel (to-device mel accel))
    (define a-batch-loss
      (hybrid-batch-loss a-net a-vocab
                         (list a-mel (narrow a-mel 1 0 200))
                         (list transcript "MISTER QUILTER")))
    (check-equal? (tensor-device a-batch-loss) accel)
    (check-true (rational? (item a-batch-loss)))
    (check-false (nan? (item a-batch-loss)))))
