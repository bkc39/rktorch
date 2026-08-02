#lang racket/base

;; End-to-end training demo for the 06-gpt capstone (#24): download Heart of
;; Darkness (cached), train the example's gpt for EPOCHS full passes over the
;; corpus blocks, print the mean training loss per epoch, then print greedy
;; samples from a few prompts to show the model working.
;;
;; Run (GPU):  nix develop .#cuda --command racket scripts/train-gpt.rkt
;; Run (CPU):  nix develop --command racket scripts/train-gpt.rkt
;;
;; EPOCHS / BATCH / BLOCK environment variables override the defaults below
;; (e.g. EPOCHS=5 for a quick look). The defaults reproduce the headline run:
;; ~40 epochs ≈ 2000 minibatch steps, loss ~4.4 -> ~1.5 on an RTX 3090 Ti in
;; about two minutes.

(require (only-in racket/format ~r)
         torch
         torch/nn
         (only-in torch/data/text
                  contiguous-blocks
                  encode
                  load-heart-of-darkness
                  text->vocab)
         (only-in "../examples/racket/06-gpt.rkt" generate gpt pick-device))

(define epochs (string->number (or (getenv "EPOCHS") "40")))
(define batch (string->number (or (getenv "BATCH") "64")))
(define block-size (string->number (or (getenv "BLOCK") "64")))
(define prompts '("The " "Marlow " "The river "))

(define device (pick-device))
(printf "device: ~a\n" device)

(with-default-device device
  (manual-seed! 0)
  (define text (load-heart-of-darkness))
  (define vocab (text->vocab text))
  (define v-size (vector-length vocab))
  (define-values (xs ys) (contiguous-blocks (encode vocab text) block-size))
  (define n (car (tensor-shape xs)))
  (printf "corpus: ~a chars, vocab ~a; ~a blocks of ~a (~a minibatches/epoch)\n"
          (string-length text) v-size n block-size (quotient n batch))
  (define net (gpt v-size block-size #:n-embd 128 #:n-head 4 #:n-layer 4))
  (define opt (adam (parameters net) #:lr 0.0003))
  ;; An epoch is one deterministic pass over the contiguous minibatches (the
  ;; trailing partial batch is dropped); the printed loss is the mean over
  ;; the epoch's steps, so early epochs average in their fast initial drop.
  (for ([epoch (in-range 1 (add1 epochs))])
    (define-values (total steps)
      (for/fold ([total 0.0] [steps 0])
                ([start (in-range 0 (- n batch) batch)])
        (zero-grads! opt)
        (define loss
          (cross-entropy
           (reshape (net (narrow xs 0 start batch)) -1 v-size)
           (reshape (narrow ys 0 start batch) -1)))
        (backward! loss)
        (step! opt)
        (values (+ total (item loss)) (add1 steps))))
    (printf "epoch ~a/~a: mean loss ~a\n"
            epoch epochs (~r (/ total steps) #:precision '(= 4)))
    (flush-output))
  (for ([prompt (in-list prompts)])
    (printf "\n--- prompt ~v ---\n~a\n" prompt
            (generate net vocab prompt
                      #:steps 300 #:block-size block-size #:device device))))
