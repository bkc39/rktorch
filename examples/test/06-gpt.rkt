#lang racket/base

;; Runner + tests for the literate ../../examples/racket/06-gpt.rkt.

(require (only-in racket/list first last)
         (only-in racket/math nan?)
         (only-in racket/string string-contains? string-prefix?)
         torch
         torch/nn
         (only-in torch/data/text encode)
         "../racket/06-gpt.rkt")

(module+ main
  (require (only-in "private/env.rkt" env-number))
  ;; EXCERPT=1: the offline middle path — train on the committed Part I
  ;; excerpt (no network), then sample. Otherwise the headline run: full
  ;; Heart of Darkness (downloads + caches), 2000 minibatch steps, then a
  ;; greedy sample; pass STEPS to override. Use run-example for the quick
  ;; offline smoke instead.
  (printf "device: ~a\n" (pick-device))
  (define-values (net vocab)
    (if (getenv "EXCERPT")
        (train-excerpt)
        (train-novel #:steps (env-number "STEPS" 2000))))
  ;; generate derives the device and context limit from the net.
  (displayln (generate net vocab "The " #:steps 400)))

(module+ test
  (require rackunit)
  ;; Deterministic, offline: 5 full-batch steps on the committed fixture.
  (define-values (losses net vocab device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  ;; The parameter tree: 2 embeddings + 2 blocks x (4 attention projections,
  ;; 2 feed-forward Linears and 2 LayerNorms, every one weight+bias) + the
  ;; stack's final norm + head = 38 tensors, under PyTorch's layer names.
  (define names (map car (named-parameters net)))
  (check-equal? (length names) 38)
  (check-equal? (first names) "tok-emb.weight")
  (check-equal? (last names) "head.bias")
  (check-not-false (member "transformer.layers.0.norm1.weight" names))
  (check-not-false
   (member "transformer.layers.0.self-attn.query.weight" names))
  (check-not-false (member "transformer.layers.1.self-attn.out.bias" names))
  (check-not-false (member "transformer.layers.1.linear2.bias" names))
  (check-not-false (member "transformer.norm.weight" names))
  ;; the hand-written blocks' names are gone
  (check-false (member "blocks.0.attention.branch.wq.weight" names))
  (define v-size (vector-length vocab))
  ;; 32 token-table + 33 head scalars per character, 25984 for the rest
  (check-equal? (for/sum ([p (in-list (parameters net))]) (numel p))
                (+ (* 65 v-size) 25984))
  ;; the embedding tables are sized by the fixture vocab and block-size 16.
  (check-equal? (shape (car (parameters net))) (list v-size 32))
  (check-equal? (tensor-shape (cadr (parameters net))) '(16 32))
  ;; the standard stack starts every block as a copy of the first, as
  ;; nn.TransformerEncoder does
  (manual-seed! 0)
  (define fresh (named-parameters (gpt v-size 16)))
  (define (block-values layer)
    (define prefix (format "transformer.layers.~a." layer))
    (for/list ([named (in-list fresh)]
               #:when (string-prefix? (car named) prefix))
      (tensor->list (cdr named))))
  (check-equal? (length (block-values 1)) 16)
  (check-equal? (block-values 1) (block-values 0))
  ;; causal: changing the last character leaves the earlier positions'
  ;; logits where they were
  (define (logits-of s)
    (with-no-grad (net (reshape (encode vocab s) 1 -1))))
  (define (prefix t) (narrow t 1 0 6))
  (check-true (< (item (max (abs (- (prefix (logits-of "The sea"))
                                    (prefix (logits-of "The set"))))))
                 1e-6))
  (check-true (> (item (max (abs (- (logits-of "The sea")
                                    (logits-of "The set")))))
                 1e-3))
  ;; generation smoke: greedy sampling appends exactly #:steps chars, stays
  ;; inside the training vocab, and leaves the net back in train mode.
  (define sample (generate net vocab "The " #:steps 20))
  (check-equal? (string-length sample) 24)
  (check-true (for/and ([c (in-string sample)])
                (and (member c (vector->list vocab)) #t))
              (format "generated chars outside the vocab: ~v" sample))
  (check-true (layer-training? net) "generate left the net in eval mode")
  (check-exn #rx"prompt must be non-empty"
             (lambda () (generate net vocab "")))
  ;; Device RNG streams differ from the CPU's, so the on-device arm checks
  ;; convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net a-vocab _a-dev) (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l))))
                        a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))
    ;; generate reads its device from the net's parameters, not the default
    (check-equal? (string-length (generate a-net a-vocab "The " #:steps 20))
                  24))
  ;; The committed Part I excerpt behind train-excerpt: data integrity only
  ;; (training it is minutes of CPU — the offline demo, not a CI job).
  (define excerpt (load-excerpt))
  (check-equal? (string-length excerpt) 30872)
  (check-true (regexp-match? #rx"^The Nellie, a cruising yawl" excerpt))
  (check-false (string-contains? excerpt "\r") "excerpt must be LF-only")
  (check-false (string-contains? excerpt "PROJECT GUTENBERG")
               "excerpt must be prose only, no PG boilerplate"))
