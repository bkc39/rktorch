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
  (displayln "wrote style-transfer.ppm"))

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

(module+ test
  ;; Parity against torch/tests/python/style_transfer.py, in the default
  ;; shell with the weights cached: the twin's pixels, VGG's activations at
  ;; the style layers, then five steps of the transfer.
  (require (only-in racket/path path-only)
           (only-in torch/tests/private/python-env
                    call-with-python-env python-check python-module-available?
                    unpack)
           (only-in torch/vision/weights
                    pretrained-weights pretrained-weights-cached?))

  (define (worst a b) (item (max (abs (sub a b)))))

  (define (check-parity)
    (define weights-dir
      (path-only (pretrained-weights 'vgg16-features-imagenet1k-v1)))
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_WEIGHTS" (path->string weights-dir)))
       (lambda () (python-check "style_transfer.py"))))
    (define their-content (unpack (hash-ref j 'content) 'float32))
    (define their-style (unpack (hash-ref j 'style) 'float32))
    (define pretrained (frozen-vgg 'cpu))
    (define found (activations pretrained their-content style-layers))
    (for ([i (in-list style-layers)])
      (define theirs
        (unpack (hash-ref (hash-ref j 'activations)
                          (string->symbol (number->string i)))
                'float32))
      (check-equal? (shape (hash-ref found i)) (shape theirs))
      (check-true (<= (worst (hash-ref found i) theirs) 1e-4)
                  (format "activations at ~a, max |difference| ~a"
                          i (worst (hash-ref found i) theirs))))
    (define-values (our-image our-losses)
      (style-transfer pretrained their-content their-style #:steps 5 #:lr 0.02))
    (define their-losses (hash-ref j 'losses))
    (check-equal? (map car our-losses) (map car their-losses))
    (for* ([(mine theirs) (in-parallel (in-list our-losses)
                                       (in-list their-losses))]
           [(a b) (in-parallel (in-list (cdr mine)) (in-list (cdr theirs)))])
      (check-true (<= (abs (- a b)) (* 1e-5 (max 1.0 (abs b))))
                  (format "step ~a: ~a against torch's ~a" (car mine) a b)))
    (define their-image (unpack (hash-ref j 'image) 'float32))
    (check-true (<= (worst our-image their-image) 1e-5)
                (format "the image after five steps, max |difference| ~a"
                        (worst our-image their-image))))

  (cond
    [(not (python-module-available? "torchvision"))
     (displayln "[16-style-transfer] parity skipped: python3 `torchvision` not available")]
    [(not (pretrained-weights-cached? 'vgg16-features-imagenet1k-v1))
     (displayln "[16-style-transfer] parity skipped: VGG-16 weights not cached")]
    [else (check-parity)]))
