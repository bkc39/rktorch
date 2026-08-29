#lang racket/base

;; Runner + tests for the literate ../../examples/racket/07-asr.rkt.

(require (only-in racket/list first last)
         (only-in racket/math nan?)
         torch
         torch/nn
         (only-in torch/audio/librispeech load-librispeech-fixture)
         (only-in torch/audio/metrics cer wer)
         "../racket/07-asr.rkt")

(module+ main
  ;; The headline run: download dev-clean (cached), train utterance-at-a-time
  ;; over the first LIMIT utterances for EPOCHS epochs, then transcribe the
  ;; committed fixture and report exact WER/CER. Use run-example for the
  ;; quick offline smoke instead.
  (printf "device: ~a\n" (pick-device))
  (define-values (net vocab)
    (train-librispeech
     #:epochs (string->number (or (getenv "EPOCHS") "10"))
     #:limit (string->number (or (getenv "LIMIT") "64"))))
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define hypothesis
    (greedy-decode net vocab (utterance-features samples rate)))
  (printf "ref: ~a\nhyp: ~a\nwer: ~a  cer: ~a\n"
          transcript hypothesis
          (wer transcript hypothesis) (cer transcript hypothesis)))

(module+ test
  (require rackunit)
  ;; Deterministic, offline: 5 CTC steps on the committed fixture.
  (define-values (losses net vocab device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  ;; The parameter tree: two Conv1d layers and the head, weight+bias each.
  (define names (map car (named-parameters net)))
  (check-equal? names
                '("conv1.weight" "conv1.bias"
                  "conv2.weight" "conv2.bias"
                  "head.weight" "head.bias"))
  (check-equal? (tensor-shape (car (parameters net))) '(64 80 3))
  (check-equal? (tensor-shape (last (parameters net)))
                (list (add1 (vector-length vocab))))
  ;; Decode smoke: a string over the training vocab (possibly empty this
  ;; early), the net left back in train mode, and the metrics computable.
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define hypothesis
    (greedy-decode net vocab (utterance-features samples rate)))
  (check-true (string? hypothesis))
  (check-true (for/and ([c (in-string hypothesis)])
                (and (member c (vector->list vocab)) #t))
              (format "decoded chars outside the vocab: ~v" hypothesis))
  (check-true (module-training? net) "greedy-decode left the net in eval mode")
  (check-true (<= 0 (wer transcript hypothesis)))
  (check-true (<= 0 (cer transcript hypothesis)))
  ;; Device RNG streams differ from the CPU's, so the on-device arm checks
  ;; convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net a-vocab _a-dev)
      (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l))))
                        a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))
    (check-true
     (string? (greedy-decode a-net a-vocab
                             (utterance-features samples rate))))))
