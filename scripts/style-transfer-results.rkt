#lang racket/base

;; Regenerates torch/scribblings/results/style-transfer.rktd and
;; style-transfer.png, the result the style-transfer example's chapter
;; shows: the honey bee photograph painted in the style of The Starry Night.
;;
;;   nix develop .#cuda --command racket scripts/style-transfer-results.rkt

(require (only-in racket/date current-date date->string date-display-format)
         (only-in racket/draw make-bitmap)
         (only-in racket/class send)
         (only-in racket/pretty pretty-write)
         racket/runtime-path
         torch
         (only-in torch/vision/transforms convert-image-dtype)
         "../examples/racket/16-style-transfer.rkt")

(define-runtime-path content-path
  "../torch/vision/fixtures/hymenoptera/bees/honey-bee.jpg")
(define-runtime-path style-path
  "../torch/vision/fixtures/style/starry-night.jpg")
(define-runtime-path results "../torch/scribblings/results")

(define size 384)
(define steps 1000)
(define lr 0.02)
(define every 50)

(define (write-png path image)
  (define-values (c h w) (apply values (shape image)))
  (define plane (* h w))
  (define argb (make-bytes (* 4 plane) 255))
  (for ([v (in-flattened-tensor
            (convert-image-dtype (to image (cpu-device)) 'uint8))]
        [k (in-naturals)])
    (define-values (channel i) (quotient/remainder k plane))
    (bytes-set! argb (+ (* 4 i) 1 channel) v))
  (define bitmap (make-bitmap w h #f))
  (send bitmap set-argb-pixels 0 0 w h argb)
  (send bitmap save-file path 'png))

(define device (pick-device))
(manual-seed! 0)
(define net (frozen-vgg device))
(define content (load-image content-path size device))
(define style (load-image style-path size device))
(define start (current-inexact-milliseconds))
(define-values (image losses)
  (style-transfer net content style #:steps steps #:lr lr))
(define seconds (/ (- (current-inexact-milliseconds) start) 1000.0))

(write-png (build-path results "style-transfer.png") (select image 0 0))
(call-with-output-file (build-path results "style-transfer.rktd")
  #:exists 'truncate
  (lambda (port)
    (pretty-write
     (hasheq 'content "hymenoptera/bees/honey-bee.jpg"
             'style "style/starry-night.jpg"
             'size size
             'steps steps
             'lr lr
             'seconds (/ (round (* 10 seconds)) 10.0)
             'device (symbol->string (device-type device))
             'torch (torch-version)
             'date (parameterize ([date-display-format 'iso-8601])
                     (date->string (current-date)))
             'losses (for/list ([record (in-list losses)]
                                #:when (or (= 1 (car record))
                                           (zero? (modulo (car record) every))))
                       record))
     port)))
(printf "~a steps in ~as on ~a\n" steps (/ (round (* 10 seconds)) 10.0) device)
