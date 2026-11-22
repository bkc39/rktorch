#lang racket/base

;; Runner + tests for the literate torch/scribblings/examples/15-finetune.rkt.

(require (only-in racket/runtime-path #%datum define-runtime-path))

(define-runtime-path photos-dir "../../torch/vision/fixtures/hymenoptera")

(module+ main
  (require (only-in racket/format ~r)
           torch/scribblings/examples/15-finetune
           (only-in "private/env.rkt" env-number))
  ;; The headline run: the tutorial's ants and bees (downloads and caches
  ;; the 47 MB archive once) on torchvision's ResNet-18 (and its 47 MB
  ;; checkpoint), both phases, validation accuracy after every epoch.
  (printf "device: ~a\n" (pick-device))
  (define-values (records _net)
    (finetune #:feature-epochs (env-number "FEATURE_EPOCHS" 10)
              #:finetune-epochs (env-number "FINETUNE_EPOCHS" 15)))
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
           torch/scribblings/examples/15-finetune
           (only-in torch/vision/image-folder image-folder))
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
  (check-equal? (shape (cdr (assoc "fc.weight" (named-parameters net))))
                '(2 512))
  (check-true (layer-training? net) "accuracy left the net in eval mode")
  (define-values (val-xs val-ys) (load-val folder 'cpu))
  (define by-hand
    (in-eval-mode net
      (with-no-grad
        (define n (car (shape val-xs)))
        (define correct
          (for/sum ([start (in-range 0 n 64)])
            (define len (min 64 (- n start)))
            (~> (narrow val-xs 0 start len) net (argmax 1) (eq (narrow val-ys 0 start len))
                sum item)))
        (exact->inexact (/ correct n)))))
  (check-= (validation-accuracy net val-xs val-ys) by-hand 6e-8
           "M17: validation-accuracy is the old helper's share")
  (define-values (exported _syntax) (module->exports 'torch/scribblings/examples/15-finetune))
  (check-false (for*/or ([phase (in-list exported)] [name (in-list (cdr phase))])
                 (eq? (car name) 'accuracy))
               "M17: 15 no longer exports accuracy")
  (set-frozen! net #t)
  (check-equal? (for/list ([(name p) (in-named-parameters net)]
                           #:when (requires-grad? p))
                  name)
                '("fc.weight" "fc.bias"))
  (set-frozen! net #f)
  (check-true (andmap requires-grad? (parameters net)))
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-records a-net) (run accel))
    (check-equal? (device (last (parameters a-net))) accel)
    (for ([r (in-list a-records)])
      (check-true (rational? (list-ref r 2))
                  (format "non-finite loss on ~a: ~a" accel r)))))
