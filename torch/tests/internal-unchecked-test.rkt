#lang racket/base

(module+ test
  (require (only-in racket/contract/base has-contract?)
           (only-in racket/match match match-define)
           rackunit
           "../data/loader.rkt"
           "../main.rkt"
           "../vision/transforms.rkt")

  (define (resolved index)
    (resolved-module-path-name (module-path-index-resolve index)))

  (define (module-name path)
    (resolved (module-path-index-join path #f)))

  (define facade (module-name 'torch/foreign))
  (define unchecked (module-name '(submod torch/foreign unchecked)))

  (define (imported-from? name source)
    (match (identifier-binding (namespace-symbol->identifier name))
      [(list _ _ nominal _ ...) (equal? (resolved nominal) source)]
      [_ #f]))

  (define (imports module source)
    (dynamic-require module #f)
    (parameterize ([current-namespace (module->namespace module)])
      (for/list ([name (in-list (namespace-mapped-symbols))]
                 #:when (imported-from? name source))
        (cons name (with-handlers ([exn:fail:syntax? (lambda (_) #f)])
                     (eval name))))))

  (define (contracted names)
    (sort (for/list ([entry (in-list names)] #:when (has-contract? (cdr entry)))
            (car entry))
          symbol<?))

  (define hot-modules
    '((torch/vision/transforms)
      (torch/data/dataset)
      (torch/data/loader draw-seed index-select make-generator randperm select)))

  (test-case "the loader's hot modules cross only the contracts they list"
    (for ([entry (in-list hot-modules)])
      (match-define (cons module allowed) entry)
      (check-equal? (contracted (imports module facade)) allowed
                    (symbol->string module))))

  (test-case "their unchecked imports are the plain definitions"
    (for ([entry (in-list hot-modules)])
      (define module (car entry))
      (define names (imports module unchecked))
      (check-not-equal? names '() (symbol->string module))
      (check-equal? (contracted names) '() (symbol->string module)))))

(module+ test
  (require (only-in racket/contract/combinator exn:fail:contract:blame?)
           rackunit
           "../data/loader.rkt"
           "../main.rkt"
           "../vision/transforms.rkt")

  (define-namespace-anchor here)

  (define (blames-this-module? e)
    (and (exn:fail:contract:blame? e)
         (regexp-match? #rx"blaming: [(][^)]*internal-unchecked-test[.]rkt test[)]"
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (define image (zeros 3 4 4))
  (define ds (tensor-dataset (ones 4 2)))

  (define entry-calls
    '((random-horizontal-flip 'image)
      (random-horizontal-flip (zeros 4 4))
      (random-crop (zeros 3 0 4))
      (resize (zeros 3 4 4 #:dtype 'int64) 2)
      (center-crop image 5)
      (normalize 'image '(0.0 0.0 0.0) '(1.0 1.0 1.0))
      (normalize image "mean" '(1.0))
      (normalize image '(0.0) '(1.0) #:inplace? 'yes)
      (normalize image '(0.0 0.0) '(1.0))
      (normalize image '(0.0) '(0.0))
      (imagenet-normalize (zeros 1 4 4))
      (convert-image-dtype (zeros 3 2 2 #:dtype 'int64))
      (imagenet-preprocess "image")
      (random-resized-crop (zeros 3 0 2) 4)
      (default-collate (list (list (ones 2)) (list (ones 3))))
      (default-collate '((image)))
      (indices->list (tensor '(1.0)))
      (dataset-batch ds (tensor '(1.0 2.0)) default-collate)
      (tensor-dataset (ones 2) (ones 3))
      (tensor-dataset (tensor 1.0))
      (dataloader 'dataset)))

  (test-case "each entry point that calls unchecked bindings blames its caller"
    (define ns (namespace-anchor->namespace here))
    (for ([call (in-list entry-calls)])
      (check-exn blames-this-module? (lambda () (eval call ns)) (format "~s" call)))))
