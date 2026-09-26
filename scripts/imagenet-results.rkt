#lang racket/base

;; Regenerates torch/scribblings/results/imagenet-top5.rktd, the table the
;; guide's pretrained chapter shows: examples/racket/14-imagenet.rkt's
;; ResNet-18 on the committed photographs.
;;
;;   nix develop .#cuda --command racket scripts/imagenet-results.rkt

(require (only-in racket/date current-date date->string date-display-format)
         (only-in racket/list last)
         (only-in racket/pretty pretty-write)
         racket/runtime-path
         torch
         (only-in torch/vision/weights pretrained-weights)
         "../examples/racket/14-imagenet.rkt")

(define-runtime-path photos-dir "../torch/vision/fixtures/hymenoptera")
(define-runtime-path out "../torch/scribblings/results/imagenet-top5.rktd")

(define photos
  (for*/list ([class (in-list '("ants" "bees"))]
              [name (in-list (sort (map path->string
                                        (directory-list
                                         (build-path photos-dir class)))
                                   string<?))])
    (list class name)))

(define device (pick-device))
(define guesses
  (run-example (for/list ([p (in-list photos)])
                 (apply build-path photos-dir p))
               #:device device))

(define weights (pretrained-weights 'resnet18-imagenet1k-v1))

(call-with-output-file out #:exists 'truncate
  (lambda (port)
    (pretty-write
     (hasheq 'model "resnet18, torchvision IMAGENET1K_V1"
             'weights (path->string (last (explode-path weights)))
             'device (format "~a" (device-type device))
             'torch (torch-version)
             'date (parameterize ([date-display-format 'iso-8601])
                     (date->string (current-date)))
             'photos
             (for/list ([p (in-list photos)] [top (in-list guesses)])
               (hasheq 'path p
                       'top5 (for/list ([g (in-list top)])
                               (cons (car g)
                                     (/ (round (* 1e4 (cdr g))) 1e4))))))
     port)))
(printf "wrote ~a\n" out)
