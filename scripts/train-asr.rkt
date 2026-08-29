#lang racket/base

;; Train the 07-asr example on LibriSpeech dev-clean, checkpoint, transcribe.
;; Run (GPU):  nix develop .#cuda --command racket scripts/train-asr.rkt
;; Run (CPU):  nix develop --command racket scripts/train-asr.rkt

(require (only-in racket/file make-directory*)
         (only-in racket/format ~r)
         (only-in racket/path path-only)
         torch
         torch/nn
         (only-in torch/audio/librispeech
                  librispeech-utterances load-utterance utterance-transcript)
         (only-in torch/audio/metrics cer wer)
         (only-in "../examples/racket/07-asr.rkt"
                  greedy-decode pick-device train-librispeech
                  utterance-features))

(define epochs (string->number (or (getenv "EPOCHS") "20")))
(define limit (string->number (or (getenv "LIMIT") "256")))
(define n-hidden (string->number (or (getenv "HIDDEN") "128")))
(define checkpoint (or (getenv "CHECKPOINT") "checkpoints/asr-dev-clean"))

(define device (pick-device))
(printf "device: ~a\n" device)

(define-values (net vocab)
  (train-librispeech #:epochs epochs #:limit limit #:n-hidden n-hidden
                     #:device device))
(reclaim-native-memory!)
(let ([dir (path-only (string->path checkpoint))])
  (when dir (make-directory* dir)))
(save-state! net (string-append checkpoint ".safetensors"))
(call-with-output-file (string-append checkpoint ".rktd") #:exists 'replace
  (lambda (out)
    (write (list (cons 'vocab (list->string (vector->list vocab)))
                 (cons 'n-mels 80)
                 (cons 'n-hidden n-hidden)
                 (cons 'epochs epochs)
                 (cons 'limit limit))
           out)))
(printf "\ncheckpoint: ~a.safetensors (+ .rktd sidecar)\n" checkpoint)

;; score a few held-out utterances (the tail of dev-clean sits past the
;; training window whenever limit < the split's utterance count)
(define all (librispeech-utterances "dev-clean"))
(for ([u (in-list (list-tail all (min limit (- (length all) 3))))]
      [_ (in-range 3)])
  (define-values (samples rate) (load-utterance u))
  (define reference (utterance-transcript u))
  (define hypothesis
    (greedy-decode net vocab (utterance-features samples rate)))
  (define w (wer reference hypothesis))
  (define c (cer reference hypothesis))
  (printf "\nref: ~a\nhyp: ~a\nwer: ~a (~a)  cer: ~a (~a)\n"
          reference hypothesis
          w (~r (exact->inexact w) #:precision '(= 3))
          c (~r (exact->inexact c) #:precision '(= 3))))
