#lang racket/base

(module+ test
  (require (only-in racket/path path-only)
           (only-in rackunit check-equal? check-true)
           (only-in "../main.rkt" abs item max sub tensor-shape)
           (only-in "../vision/weights.rkt"
                    pretrained-weights pretrained-weights-cached?)
           (only-in "../../examples/racket/16-style-transfer.rkt"
                    activations frozen-vgg style-layers style-transfer)
           "private/python-env.rkt")

  (define (worst a b) (item (max (abs (sub a b)))))

  (cond
    [(not (python-module-available? "torchvision"))
     (displayln
      "[style-transfer-parity-test] skipped: python3 `torchvision` not available")]
    [(not (pretrained-weights-cached? 'vgg16-features-imagenet1k-v1))
     (printf "[style-transfer-parity-test] skipped: VGG-16 weights not cached ~a\n"
             "(pretrained-weights fetches them)")]
    [else
     (define weights-dir
       (path-only (pretrained-weights 'vgg16-features-imagenet1k-v1)))
     (define j
       (call-with-python-env
        #:env (list (cons "RKTORCH_PARITY_WEIGHTS" (path->string weights-dir)))
        (lambda () (python-check "style_transfer.py"))))
     (define content (unpack (hash-ref j 'content) 'float32))
     (define style (unpack (hash-ref j 'style) 'float32))
     (define net (frozen-vgg 'cpu))

     (define found (activations net content style-layers))
     (for ([i (in-list style-layers)])
       (define theirs
         (unpack (hash-ref (hash-ref j 'activations) (string->symbol (number->string i)))
                 'float32))
       (check-equal? (tensor-shape (hash-ref found i)) (tensor-shape theirs))
       (check-true (<= (worst (hash-ref found i) theirs) 1e-4)
                   (format "activations at ~a, max |difference| ~a"
                           i (worst (hash-ref found i) theirs))))

     (define-values (image losses)
       (style-transfer net content style #:steps 5 #:lr 0.02))
     (check-equal? (map car losses) (map car (hash-ref j 'losses)))
     (for* ([(ours theirs) (in-parallel (in-list losses)
                                        (in-list (hash-ref j 'losses)))]
            [(a b) (in-parallel (in-list (cdr ours)) (in-list (cdr theirs)))])
       (check-true (<= (abs (- a b)) (* 1e-5 (max 1.0 (abs b))))
                   (format "step ~a: ~a against torch's ~a" (car ours) a b)))
     (define their-image (unpack (hash-ref j 'image) 'float32))
     (check-true (<= (worst image their-image) 1e-5)
                 (format "the image after five steps, max |difference| ~a"
                         (worst image their-image)))]))
