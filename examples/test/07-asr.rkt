#lang racket/base

(require (only-in racket/list first last)
         (only-in racket/math nan?)
         torch
         torch/nn
         (only-in torch/audio/librispeech load-librispeech-fixture)
         (only-in torch/audio/metrics cer wer)
         "../racket/07-asr.rkt")

(module+ main
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
  (define-values (losses net vocab device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses)
              (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (define names (map car (named-parameters net)))
  (check-equal? (length names) 273)
  (check-equal? (first names) "conv1.weight")
  (check-equal? (last names) "head.bias")
  (check-not-false (member "dil4.bias" names))
  (check-not-false (member "enc6.wq.weight" names))
  (check-not-false (member "tok-emb.weight" names))
  (check-not-false (member "dec6.co.bias" names))
  (check-equal? (tensor-shape (car (parameters net))) '(64 80 3))
  (check-equal? (tensor-shape
                 (cdr (assoc "tok-emb.weight" (named-parameters net))))
                (list (+ (vector-length vocab) 2) 64))
  (define-values (samples rate transcript) (load-librispeech-fixture))
  (define features (utterance-features samples rate))
  (define ctc-hyp (greedy-decode net vocab features))
  (define att-hyp (transcribe net vocab features))
  (for ([hypothesis (in-list (list ctc-hyp att-hyp))])
    (check-true (string? hypothesis))
    (check-true (for/and ([c (in-string hypothesis)])
                  (and (member c (vector->list vocab)) #t))
                (format "decoded chars outside the vocab: ~v" hypothesis)))
  (check-true (module-training? net) "decoding left the net in eval mode")
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
  (check-true (<= (string-length att-hyp) 500)
              (format "transcribe failed to terminate: ~v" att-hyp))
  (check-true (device? (pick-device)))
  ;; Device RNG streams differ from the CPU's, so the on-device arm checks
  ;; convergence, never equality with the CPU losses above. pick-device
  ;; already excludes MPS (no ctc_loss kernel), so this arm is CUDA-only.
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
    (check-true (string? (transcribe a-net a-vocab features)))))
