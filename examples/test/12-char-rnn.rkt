#lang racket/base

(require (only-in racket/list first last)
         (only-in racket/math nan?)
         torch
         torch/nn
         "../racket/12-char-rnn.rkt")

(module+ main
  (printf "device: ~a\n" (pick-device))
  ;; an unset EPOCHS takes the default; a supplied one is authoritative, zero
  ;; included, so EPOCHS=0 samples an untrained net rather than silently
  ;; starting the full run and downloading the novella for it
  (define epochs
    (let ([supplied (getenv "EPOCHS")])
      (and supplied
           (or (string->number supplied)
               (error '12-char-rnn "EPOCHS is not a number: ~a" supplied)))))
  (define-values (net vocab)
    (cond
      [(and (getenv "EXCERPT") epochs) (train-excerpt #:epochs epochs)]
      [(getenv "EXCERPT") (train-excerpt)]
      [epochs (train-novel #:epochs epochs)]
      [else (train-novel)]))
  (manual-seed! (string->number (or (getenv "SEED") "0")))
  (displayln
   (sample net vocab "The "
           #:steps 600
           #:temperature (string->number (or (getenv "TEMPERATURE") "0.8")))))

(module+ test
  (require rackunit)
  (define (finite? l) (and (rational? l) (not (nan? l))))
  (define-values (losses net vocab device) (run-example #:device 'cpu))
  (check-equal? device 'cpu)
  (check-equal? (length losses) 5)
  (check-true (andmap finite? losses) (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (check-equal? (map car (named-parameters net))
                '("embed.weight"
                  "lstm.weight_ih_l0" "lstm.weight_hh_l0"
                  "lstm.bias_ih_l0" "lstm.bias_hh_l0"
                  "head.weight" "head.bias"))
  (check-equal? (tensor-shape (car (parameters net)))
                (list (vector-length vocab) 32))
  (define-values (logits state) (net (to-dtype (tensor '((0 1 2))) 'int64) #f))
  (check-equal? (tensor-shape logits) (list 1 3 (vector-length vocab)))
  (check-equal? (map tensor-shape state) '((1 1 64) (1 1 64)))
  (define (seeded-sample temperature)
    (manual-seed! 7)
    (sample net vocab "The " #:steps 20 #:temperature temperature))
  (define text (seeded-sample 0.8))
  (check-equal? (string-length text) 24)
  (check-equal? (substring text 0 4) "The ")
  (check-true (for/and ([c (in-string text)])
                (and (member c (vector->list vocab)) #t))
              (format "sampled chars outside the vocab: ~v" text))
  (check-equal? (seeded-sample 0.8) text "a seed makes the sample repeatable")
  (check-true (layer-training? net) "sample left the net in eval mode")
  (check-exn #rx"prompt must be non-empty" (lambda () (sample net vocab "")))
  (check-exn #rx"temperature must be positive"
             (lambda () (sample net vocab "The " #:temperature 0)))
  (define deep (char-rnn (vector-length vocab) #:num-layers 2 #:dropout 0.2))
  (check-equal? (string-length (sample deep vocab "It " #:steps 5)) 8)
  ;; Device RNG streams differ from the CPU's, so the on-device arm checks
  ;; convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net a-vocab _a-dev) (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap finite? a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))
    (check-equal? (string-length (sample a-net a-vocab "The " #:steps 20)) 24)))
