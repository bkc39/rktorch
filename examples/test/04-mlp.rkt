#lang racket/base

;; Runner + tests for the literate example ../../examples/racket/04-mlp.rkt.

(require (except-in racket/list argmax flatten)
         torch
         torch/nn
         "../racket/04-mlp.rkt")

(module+ main
  (define-values (losses net device) (run-example))
  (printf "device: ~a\n" device)
  (printf "losses: ~a\n" losses)
  (for ([nm+p (in-list (named-parameters net))])
    (printf "~a: ~a\n" (car nm+p) (tensor-shape (cdr nm+p)))))

(module+ test
  (require rackunit)
  (define-values (losses net device) (run-example))
  ;; run-example echoes back the device it ran on; the no-argument call gets
  ;; pick-device's normalised struct, whichever accelerator this host has.
  (check-not-false (and (device? device)
                        (memq (device-type device) '(cpu cuda mps)))
                   (format "unexpected device: ~a" device))
  (check-equal? (length losses) 5)
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))
  (check-equal? (map car (named-parameters net))
                '("fc1.weight" "fc1.bias" "fc2.weight" "fc2.bias"))
  (check-equal? (map tensor-shape (parameters net))
                '((8 4) (8) (2 8) (2))))
