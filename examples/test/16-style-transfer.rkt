#lang racket/base

;; Runner + tests for the literate ../../examples/racket/16-style-transfer.rkt.

(require racket/runtime-path)

(define-runtime-path content-path
  "../../torch/vision/fixtures/hymenoptera/bees/honey-bee.jpg")
(define-runtime-path style-path
  "../../torch/vision/fixtures/style/starry-night.jpg")

(module+ main
  (require (only-in racket/format ~r)
           (only-in torch select)
           (only-in torch/vision/ppm write-ppm)
           "../racket/16-style-transfer.rkt")
  ;; The headline run: torchvision's VGG-16 features (a 56 MiB range of
  ;; the checkpoint, fetched once), 1000 Adam steps at 384 pixels, the
  ;; result written beside the working directory.
  (define device (pick-device))
  (printf "device: ~a\n" device)
  (define net (frozen-vgg device))
  (define-values (image losses)
    (style-transfer net (load-image content-path 384 device)
                    (load-image style-path 384 device)
                    #:steps 1000))
  (for ([record (in-list losses)]
        #:when (zero? (modulo (car record) 100)))
    (printf "step ~a: style ~a, content ~a\n" (car record)
            (~r (cadr record) #:notation 'exponential #:precision 3)
            (~r (caddr record) #:precision 2))
    (flush-output))
  (write-ppm "style-transfer.ppm" (select image 0 0))
  (printf "wrote style-transfer.ppm\n"))

(module+ test
  (require (only-in racket/list last)
           rackunit
           torch
           (only-in torch/nn parameters)
           "../racket/16-style-transfer.rkt")
  ;; Offline and deterministic: random weights, both images at 32 pixels,
  ;; five steps on the CPU.
  (manual-seed! 0)
  (define device (cpu-device))
  (define net (frozen-vgg device #:pretrained? #f))
  (check-true (for/and ([p (in-list (parameters net))])
                (not (requires-grad? p)))
              "the network is frozen")
  (define content (load-image content-path 32 device))
  (define style (load-image style-path 32 device))
  (check-equal? (shape content) '(1 3 32 40))
  (define found (activations net content (cons content-layer style-layers)))
  (check-equal? (sort (hash-keys found) <) '(0 2 5 7 10))
  (check-equal? (shape (hash-ref found 10)) '(1 256 8 10))
  (check-equal? (shape (gram-matrix (hash-ref found 10))) '(256 256))
  (define-values (image losses)
    (style-transfer net content style #:steps 5 #:lr 0.05))
  (check-equal? (map car losses) '(1 2 3 4 5))
  (check-equal? (shape image) (shape content))
  (check-false (requires-grad? image) "the result is detached")
  (check-true (<= 0.0 (item (min image)) (item (max image)) 1.0)
              "the pixels stay in [0, 1]")
  (check-true (< (cadr (last losses)) (cadr (car losses)))
              "the style loss falls")
  (check-equal? (caddr (car losses)) 0.0
                "the image starts as the photograph"))
