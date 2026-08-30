#lang racket/base

;; Run (GPU):  nix develop .#cuda --command racket scripts/train-asr.rkt
;; Run (CPU):  nix develop --command racket scripts/train-asr.rkt

(require (only-in racket/file make-directory*)
         (only-in racket/format ~r)
         (only-in racket/path path-only)
         torch
         torch/nn
         (only-in torch/audio/librispeech
                  librispeech-utterances load-utterance utterance-transcript)
         (only-in "../examples/racket/07-asr.rkt"
                  evaluate greedy-decode pick-device train-librispeech
                  transcribe utterance-features))

(define epochs (string->number (or (getenv "EPOCHS") "20")))
(define limit
  (let ([l (getenv "LIMIT")])
    (and l (or (string->number l)
               (error 'train-asr "LIMIT is not a number: ~a" l)))))
(define batch (string->number (or (getenv "BATCH") "16")))
(define n-embd (string->number (or (getenv "EMBD") "256")))
(define checkpoint (or (getenv "CHECKPOINT") "checkpoints/asr-dev-clean"))
(define held-out 3)

(when (and limit (not (exact-positive-integer? limit)))
  (error 'train-asr "LIMIT must be a positive integer: ~a" limit))

(define device (pick-device))
(printf "device: ~a\n" device)

(define all (librispeech-utterances "dev-clean"))
(define train-limit
  (min (or limit (- (length all) held-out)) (- (length all) held-out)))
(define-values (net vocab)
  (train-librispeech #:epochs epochs #:limit train-limit #:batch batch
                     #:n-embd n-embd #:device device))
(reclaim-native-memory!)
(let ([dir (path-only (string->path checkpoint))])
  (when dir (make-directory* dir)))
(save-state! net (string-append checkpoint ".safetensors"))
(call-with-output-file (string-append checkpoint ".rktd") #:exists 'replace
  (lambda (out)
    (write (list (cons 'vocab (list->string (vector->list vocab)))
                 (cons 'n-mels 80)
                 (cons 'n-embd n-embd)
                 (cons 'n-head 4)
                 (cons 'epochs epochs)
                 (cons 'batch batch)
                 (cons 'limit train-limit))
           out)))
(printf "\ncheckpoint: ~a.safetensors (+ .rktd sidecar)\n" checkpoint)

(define held (list-tail all (- (length all) held-out)))

(for ([u (in-list held)])
  (define-values (samples rate) (load-utterance u))
  (define features (utterance-features samples rate))
  (printf "\nref:     ~a\nctc:     ~a\nattend:  ~a\n"
          (utterance-transcript u)
          (greedy-decode net vocab features)
          (transcribe net vocab features)))

(define-values (corpus-wer corpus-cer) (evaluate net vocab held))
(printf "\nheld-out (~a utterances): wer ~a  cer ~a\n"
        held-out
        (~r (exact->inexact corpus-wer) #:precision '(= 4))
        (~r (exact->inexact corpus-cer) #:precision '(= 4)))
