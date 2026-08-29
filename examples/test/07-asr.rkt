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
  ;; over the whole split (or the first LIMIT utterances) for EPOCHS epochs,
  ;; then transcribe the committed fixture both ways and report exact
  ;; WER/CER. Use run-example for the quick offline smoke instead.
  (printf "device: ~a\n" (pick-device))
  (define limit (getenv "LIMIT"))
  (define-values (net vocab)
    (train-librispeech
     #:epochs (string->number (or (getenv "EPOCHS") "20"))
     #:limit (and limit (string->number limit))))
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define features (utterance-features samples rate))
  (define hypothesis (transcribe net vocab features))
  (printf "ref:     ~a\nctc:     ~a\nattend:  ~a\nwer: ~a  cer: ~a\n"
          transcript (greedy-decode net vocab features) hypothesis
          (wer transcript hypothesis) (cer transcript hypothesis)))

(module+ test
  (require rackunit)
  ;; Deterministic, offline: 5 hybrid CTC+CE steps on the committed fixture.
  (define-values (losses net vocab device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  ;; The parameter tree: 5 convs + 2 encoder blocks (8 tensors each) +
  ;; encoder norm + ctc head + token table + 2 decoder blocks (13 params,
  ;; 26 tensors) + decoder norm + attention head.
  (define names (map car (named-parameters net)))
  (check-equal? (length names) 103)
  (check-equal? (first names) "conv1.weight")
  (check-equal? (last names) "head.bias")
  (check-not-false (member "dil3.bias" names))
  (check-not-false (member "enc1.wq.weight" names))
  (check-not-false (member "tok-emb.weight" names))
  (check-not-false (member "dec2.co.bias" names))
  (check-equal? (tensor-shape (car (parameters net))) '(64 80 3))
  (check-equal? (tensor-shape
                 (cdr (assoc "tok-emb.weight" (named-parameters net))))
                (list (+ (vector-length vocab) 2) 64))
  ;; Decode smoke, both paths: strings over the training vocab (possibly
  ;; empty this early), the net left back in train mode, metrics computable.
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define features (utterance-features samples rate))
  (define ctc-hyp (greedy-decode net vocab features))
  (define att-hyp (transcribe net vocab features))
  (for ([hypothesis (in-list (list ctc-hyp att-hyp))])
    (check-true (string? hypothesis))
    (check-true (for/and ([c (in-string hypothesis)])
                  (and (member c (vector->list vocab)) #t))
                (format "decoded chars outside the vocab: ~v" hypothesis))
    (check-true (<= 0 (wer transcript hypothesis)))
    (check-true (<= 0 (cer transcript hypothesis))))
  (check-true (module-training? net) "decoding left the net in eval mode")
  ;; the attention decoder is capped at one character per encoder frame
  (check-true (<= (string-length att-hyp) 500)
              (format "transcribe failed to terminate: ~v" att-hyp))
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
    (check-true (string? (transcribe a-net a-vocab features)))))
