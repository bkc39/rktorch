#lang racket/base

(require (only-in racket/list first last [take list-take])
         (only-in racket/math nan?)
         torch
         torch/nn
         torch/data/translation
         "../racket/13-translation.rkt")

(module+ main
  (printf "device: ~a\n" (pick-device))
  (define-values (net source-vocab target-vocab held-out)
    (train-translator
     #:epochs (string->number (or (getenv "EPOCHS") "30"))))
  (printf "held-out token error rate over ~a pairs: ~a\n"
          (length held-out)
          (token-error-rate net source-vocab target-vocab held-out))
  (define shown (list-take held-out 10))
  (for ([pair (in-list shown)]
        [english (in-list (translate net source-vocab target-vocab
                                     (map car shown)))])
    (printf "> ~a\n= ~a\n< ~a\n\n" (car pair) (cdr pair) english)))

(module+ test
  (require rackunit)
  (define (finite? l) (and (rational? l) (not (nan? l))))
  (define-values (losses net source-vocab target-vocab)
    (run-example #:device 'cpu))
  (check-equal? (length losses) 5)
  (check-true (andmap finite? losses) (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (define names (map car (named-parameters net)))
  (check-equal? (length names) 18)
  (check-equal? (first names) "enc.embed.weight")
  (check-not-false (member "enc.gru.weight_ih_l0" names))
  (check-not-false (member "dec.attend.va.bias" names))
  (check-not-false (member "dec.gru.bias_hh_l0" names))
  (check-equal? (last names) "dec.head.bias")
  (define-values (sources targets)
    (pairs->tensors (list-take (load-translation-fixture) 4)
                    source-vocab target-vocab #:width 10))
  (define v (vocab-size target-vocab))
  (check-equal? (tensor-shape (net sources targets 10 #t)) (list 4 10 v))
  (check-equal? (tensor-shape (net sources #f 10 #f)) (list 4 10 v))
  (define translations
    (translate net source-vocab target-vocab '("Je vais bien." "Il est là !")))
  (check-equal? (length translations) 2)
  (check-true (andmap string? translations))
  (check-true (layer-training? net) "translate left the net in eval mode")
  (define rate
    (token-error-rate net source-vocab target-vocab
                      (list-take (load-translation-fixture) 8)))
  (check-true (and (rational? rate) (>= rate 0.0)))
  (define pairs (load-translation-fixture))
  (define-values (training held-out) (split-pairs pairs))
  (check-equal? (length held-out) 28)
  (check-equal? (+ (length training) (length held-out)) (length pairs))
  (define-values (_training-again held-out-again) (split-pairs pairs))
  (check-equal? held-out-again held-out)
  (define trained
    (train-pairs pairs source-vocab target-vocab
                 #:epochs 40 #:batch 32 #:hidden 64 #:lr 0.005
                 #:device 'cpu #:log-every 1000))
  (define trained-rate
    (token-error-rate trained source-vocab target-vocab pairs))
  (check-true (< trained-rate 0.5)
              (format "token error rate after training: ~a" trained-rate))
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-net a-source a-target)
      (run-example #:device accel))
    (check-equal? (tensor-device (car (parameters a-net))) accel)
    (check-true (andmap finite? a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses))
    (check-equal? (length (translate a-net a-source a-target '("je vais bien")))
                  1)))
