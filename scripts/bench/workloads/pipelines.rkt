#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/file file->string)
         (only-in racket/match match match-define match-lambda)
         (only-in racket/runtime-path define-runtime-path)
         (only-in torch
                  + * / < narrow rand select shape tensor to-dtype unsqueeze where ~>)
         (only-in torch/data/loader define-dataset)
         (only-in torch/text/data encode text->vocab)
         (only-in torch/vision/cifar10 load-cifar10-fixture)
         (only-in torch/vision/hymenoptera hymenoptera-root)
         (only-in torch/vision/image read-image)
         (only-in torch/vision/image-folder image-folder-samples)
         (only-in torch/vision/transforms
                  convert-image-dtype imagenet-mean imagenet-std normalize
                  random-crop random-horizontal-flip random-resized-crop))

(provide cifar-augment
         cifar-sources
         pipeline
         pipeline-names)

(define-runtime-path text-fixture "../../../torch/text/fixtures/heart-of-darkness-excerpt.txt")

(define pipeline-names '(jpeg cifar text))

(define cifar-mean '(0.4914 0.4822 0.4465))
(define cifar-std '(0.2470 0.2435 0.2616))
(define window 128)

(define (flip-one x)
  (~> x (unsqueeze 0) random-horizontal-flip (select 0 0)))

(define (jpeg-augment image)
  (~> image
      convert-image-dtype
      (random-resized-crop 224)
      flip-one
      (normalize imagenet-mean imagenet-std)))

(define (cifar-augment pixels)
  (~> pixels
      (to-dtype 'float32)
      (/ 255.0)
      (unsqueeze 0)
      (random-crop #:padding 4)
      random-horizontal-flip
      (select 0 0)
      (normalize cifar-mean cifar-std)))

(define (text-window ids start mask-id)
  (define span (narrow ids 0 start (add1 window)))
  (values (where (< (rand window) 0.15) mask-id (narrow span 0 0 window))
          (narrow span 0 1 window)))

(define (jpeg-files)
  (~> (build-path (hymenoptera-root) "train") image-folder-samples list->vector))

(define (cifar-sources)
  (define-values (xs ys) (load-cifar10-fixture))
  (values (~> xs (+ 1.0) (* 127.5) (+ 0.5) (to-dtype 'uint8)) ys))

(define (text-sources)
  (define text (file->string text-fixture))
  (define vocab (text->vocab text))
  (values (encode vocab text) (vector-length vocab)))

(define-dataset jpeg-cycle (files labels n)
  #:length n
  #:ref (i)
  (define k (modulo i (vector-length files)))
  (match-define (cons path _label) (vector-ref files k))
  (values (~> path (read-image #:mode 'rgb) jpeg-augment)
          (select labels 0 k)))

(define-dataset cifar-cycle (images labels n)
  #:length n
  #:ref (i)
  (define j (modulo i (car (shape images))))
  (values (cifar-augment (select images 0 j)) (select labels 0 j)))

(define-dataset masked-text (ids mask-id n)
  #:length n
  #:ref (_i)
  (text-window ids (~> (shape ids) car (- window) random) mask-id))

(define (jpeg-labels files)
  (tensor (map (match-lambda [(cons _path label) label]) (vector->list files))))

;; a pipeline's dataset of `samples` items, or its own count, and its batch size
(define (pipeline name #:samples [samples #f])
  (define (n default) (or samples default))
  (match name
    ['jpeg
     (define files (jpeg-files))
     (values (jpeg-cycle files (jpeg-labels files) (n 2048)) 32)]
    ['cifar
     (define-values (images labels) (cifar-sources))
     (values (cifar-cycle images labels (n 8192)) 128)]
    ['text
     (define-values (ids mask-id) (text-sources))
     (values (masked-text ids mask-id (n 8192)) 64)]))
