#lang racket/base

(module+ test
  (require (only-in racket/contract/base has-contract?)
           (only-in racket/contract/combinator exn:fail:contract:blame?)
           (only-in racket/match == match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           "../data/loader.rkt"
           "../main.rkt"
           "../vision/transforms.rkt")

  (define-runtime-path torch-dir "..")

  (define hot-modules
    '(("vision/transforms.rkt")
      ("data/dataset.rkt" length)
      ("data/loader.rkt" draw-seed index-select length make-generator randperm select)))

  (define (requires file)
    (match-define `(module ,_ ,_ (#%module-begin ,forms ...))
      (parameterize ([read-accept-reader #t] [read-accept-lang #t])
        (call-with-input-file (build-path torch-dir file) read)))
    (for*/list ([form (in-list forms)]
                [spec (in-list (match form [`(require ,specs ...) specs] [_ '()]))])
      spec))

  (define (imports file source)
    (for*/list ([spec (in-list (requires file))]
                [name (in-list (match spec
                                 [`(only-in ,(== source) ,names ...) names]
                                 [_ '()]))])
      name))

  (define (contracted? module name)
    (with-handlers ([exn:fail:syntax? (lambda (_) #f)])
      (has-contract? (dynamic-require module name))))

  (test-case "the loader's hot modules cross only the contracts they list"
    (for ([entry (in-list hot-modules)])
      (match-define (cons file allowed) entry)
      (check-equal? (sort (filter (lambda (name) (contracted? 'torch/foreign name))
                                  (imports file "../foreign.rkt"))
                          symbol<?)
                    allowed
                    file)))

  (test-case "their unchecked imports are the plain definitions"
    (for ([entry (in-list hot-modules)])
      (match-define (cons file _) entry)
      (define names (imports file '(submod "../foreign.rkt" unchecked)))
      (check-not-equal? names '() file)
      (for ([name (in-list names)])
        (check-false (contracted? '(submod torch/foreign unchecked) name)
                     (format "~a: ~a" file name))))
    (check-true (contracted? 'torch/foreign 'shape))
    (check-true (contracted? 'torch/foreign 'sub)))

  (define (blames-this-module? e)
    (and (exn:fail:contract:blame? e)
         (regexp-match? #rx"blaming: [(][^)]*internal-unchecked-test[.]rkt test[)]"
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (define image (zeros 3 4 4))
  (define ds (tensor-dataset (ones 4 2)))

  (define entry-points
    `((random-horizontal-flip ,(lambda () (random-horizontal-flip 'image)))
      (random-horizontal-flip ,(lambda () (random-horizontal-flip (zeros 4 4))))
      (random-crop ,(lambda () (random-crop (zeros 3 0 4))))
      (resize ,(lambda () (resize (zeros 3 4 4 #:dtype 'int64) 2)))
      (center-crop ,(lambda () (center-crop image 5)))
      (normalize ,(lambda () (normalize 'image '(0.0 0.0 0.0) '(1.0 1.0 1.0))))
      (imagenet-normalize ,(lambda () (imagenet-normalize (zeros 1 4 4))))
      (convert-image-dtype ,(lambda () (convert-image-dtype (zeros 3 2 2 #:dtype 'int64))))
      (imagenet-preprocess ,(lambda () (imagenet-preprocess "image")))
      (random-resized-crop ,(lambda () (random-resized-crop (zeros 3 0 2) 4)))
      (default-collate ,(lambda () (default-collate (list (list (ones 2)) (list (ones 3))))))
      (default-collate ,(lambda () (default-collate '((image)))))
      (indices->list ,(lambda () (indices->list (tensor '(1.0)))))
      (dataset-batch ,(lambda () (dataset-batch ds (tensor '(1.0 2.0)) default-collate)))
      (tensor-dataset ,(lambda () (tensor-dataset (ones 2) (ones 3))))
      (tensor-dataset ,(lambda () (tensor-dataset (tensor 1.0))))
      (dataloader ,(lambda () (dataloader 'dataset)))))

  (test-case "each entry point that calls unchecked bindings blames its caller"
    (for ([entry (in-list entry-points)])
      (match-define (list name thunk) entry)
      (check-exn blames-this-module? thunk (symbol->string name)))))
