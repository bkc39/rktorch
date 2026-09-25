#lang racket/base

;; Runner + tests for the literate ../../examples/racket/15-finetune.rkt.

(require racket/runtime-path)

(define-runtime-path photos-dir "../../torch/vision/fixtures/hymenoptera")

(module+ main
  (require (only-in racket/format ~r)
           "../racket/15-finetune.rkt")
  ;; The headline run: the tutorial's ants and bees (downloads and caches
  ;; the 47 MB archive once) on torchvision's ResNet-18 (and its 47 MB
  ;; checkpoint), both phases, validation accuracy after every epoch.
  (printf "device: ~a\n" (pick-device))
  (define-values (records _net) (finetune))
  (for ([r (in-list records)])
    (printf "~a epoch ~a: loss ~a, val acc ~a, ~as\n"
            (list-ref r 0) (list-ref r 1)
            (~r (list-ref r 2) #:precision '(= 4))
            (~r (list-ref r 3) #:precision '(= 4))
            (~r (list-ref r 4) #:precision '(= 1)))
    (flush-output)))

(module+ test
  (require (only-in racket/list first last)
           (only-in racket/math nan?)
           rackunit
           torch
           torch/nn
           (only-in torch/vision/image-folder image-folder)
           "../racket/15-finetune.rkt")
  ;; Offline and deterministic: random weights, the four fixture photos as
  ;; both sets, one epoch per phase in batches of two, on the CPU.
  (define folder (image-folder photos-dir))
  (define (run device)
    (finetune #:train folder #:val folder #:pretrained? #f
              #:feature-epochs 1 #:finetune-epochs 1 #:batch 2
              #:device device))
  (define-values (records net) (run 'cpu))
  (check-equal? (map (lambda (r) (list (first r) (cadr r))) records)
                '((feature-extract 1) (fine-tune 1)))
  (for ([r (in-list records)])
    (check-true (and (rational? (list-ref r 2)) (not (nan? (list-ref r 2))))
                (format "non-finite loss: ~a" r))
    (check-true (<= 0.0 (list-ref r 3) 1.0)))
  (check-equal? (tensor-shape (cdr (assoc "fc.weight" (named-parameters net))))
                '(2 512))
  (check-true (layer-training? net) "accuracy left the net in eval mode")
  (set-frozen! net #t)
  (check-equal? (for/list ([np (in-list (named-parameters net))]
                           #:when (requires-grad? (cdr np)))
                  (car np))
                '("fc.weight" "fc.bias"))
  (set-frozen! net #f)
  (check-true (andmap requires-grad? (parameters net)))
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-records a-net) (run accel))
    (check-equal? (tensor-device (last (parameters a-net))) accel)
    (for ([r (in-list a-records)])
      (check-true (rational? (list-ref r 2))
                  (format "non-finite loss on ~a: ~a" accel r)))))
