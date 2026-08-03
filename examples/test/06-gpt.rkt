#lang racket/base

;; Runner + tests for the literate ../../examples/racket/06-gpt.rkt.

(require (only-in racket/list first last)
         (only-in racket/math nan?)
         torch
         torch/nn
         "../racket/06-gpt.rkt")

(module+ main
  ;; The headline run: full Heart of Darkness (downloads + caches), 2000
  ;; minibatch steps, then a greedy sample. Pass STEPS to override. Use
  ;; run-example for the quick offline smoke instead.
  (define steps (string->number (or (getenv "STEPS") "2000")))
  (printf "device: ~a\n" (pick-device))
  (define-values (net vocab) (train-novel #:steps steps))
  ;; generate derives the device and 64-char context limit from the net.
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
  ;; The parameter tree: 2 embeddings + 2 blocks x (2 LayerNorms + 4
  ;; attention Linears + 2 MLP Linears, each weight+bias) + final ln + head
  ;; = 38 tensors, threaded through Sequential's indexed dotted names.
  (define names (map car (named-parameters net)))
  (check-equal? (length names) 38)
  (check-equal? (first names) "tok-emb.weight")
  (check-equal? (last names) "head.bias")
  (check-not-false (member "blocks.0.ln1.weight" names))
  (check-not-false (member "blocks.1.fc2.bias" names))
  ;; the embedding tables are sized by the fixture vocab and block-size 16.
  (check-equal? (tensor-shape (car (parameters net)))
                (list (vector-length vocab) 32))
  (check-equal? (tensor-shape (cadr (parameters net))) '(16 32))
  ;; generation smoke: greedy sampling appends exactly #:steps chars, stays
  ;; inside the training vocab, and leaves the net back in train mode.
  (define sample (generate net vocab "The " #:steps 20))
  (check-equal? (string-length sample) 24)
  (check-true (for/and ([c (in-string sample)])
                (and (member c (vector->list vocab)) #t))
              (format "generated chars outside the vocab: ~v" sample))
  (check-true (module-training? net) "generate left the net in eval mode"))
