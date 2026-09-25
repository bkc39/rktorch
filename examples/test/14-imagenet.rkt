#lang racket/base

;; Runner + tests for the literate ../../examples/racket/14-imagenet.rkt.

(require racket/runtime-path)

(define-runtime-path photos-dir "../../torch/vision/fixtures/hymenoptera")

(define (fixture-photos)
  (for*/list ([class (in-list '("ants" "bees"))]
              [name (in-list (sort (map path->string
                                        (directory-list
                                         (build-path photos-dir class)))
                                   string<?))])
    (build-path photos-dir class name)))

(module+ main
  (require (only-in racket/format ~r)
           "../racket/14-imagenet.rkt")
  ;; The headline run: torchvision's ResNet-18 (fetches the 47 MB
  ;; checkpoint once) on the paths given, or on the fixture photographs.
  (define args (vector->list (current-command-line-arguments)))
  (define paths (if (null? args) (fixture-photos) args))
  (printf "device: ~a\n" (pick-device))
  (for ([path (in-list paths)]
        [guesses (in-list (run-example paths))])
    (printf "~a\n" path)
    (for ([guess (in-list guesses)])
      (printf "  ~a%  ~a\n" (~r (* 100 (cdr guess)) #:precision '(= 1))
              (car guess)))))

(module+ test
  (require (only-in racket/list first)
           rackunit
           (only-in torch tensor-shape)
           (only-in torch/vision/weights pretrained-weights-cached?)
           "../racket/14-imagenet.rkt")
  ;; Offline: random weights on the CPU, so only the plumbing is checked.
  (define photos (fixture-photos))
  (define batch (photos->batch photos #:device 'cpu))
  (check-equal? (tensor-shape batch) '(4 3 224 224))
  (define tops (run-example photos #:pretrained? #f #:device 'cpu))
  (check-equal? (length tops) 4)
  (for ([guesses (in-list tops)])
    (check-equal? (length guesses) 5)
    (check-true (andmap string? (map car guesses)))
    (define ps (map cdr guesses))
    (check-true (apply >= ps) "the five come most likely first")
    (check-true (<= 0 (apply + ps) 1.000001)))
  ;; With the weights cached, the network sees the insects: each photo's
  ;; class is among its five guesses, and first for all but the close-up.
  (when (pretrained-weights-cached? 'resnet18-imagenet1k-v1)
    (define pretrained (run-example photos #:device 'cpu))
    (for ([guesses (in-list pretrained)]
          [insect (in-list '("ant" "ant" "bee" "bee"))])
      (check-not-false (assoc insect guesses) insect))
    (check-equal? (map (lambda (guesses) (car (first guesses))) pretrained)
                  '("ant" "centipede" "bee" "bee"))))
