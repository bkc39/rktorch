#lang racket/base

(module+ test
  (require (only-in racket/contract/combinator exn:fail:contract:blame?)
           (only-in racket/file file->value)
           (only-in racket/match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           "../main.rkt"
           "../vision/transforms.rkt"
           (only-in "private/op-calls.rkt" op-calls-of))

  (define-runtime-path goldens-path "../vision/fixtures/convert-normalize.rktd")

  (match-define `((bytes ,byte-shape ,byte-values)
                  (image ,image-shape ,image-values)
                  (batch ,batch-shape ,batch-values)
                  (convert ,converts ...)
                  (normalize ,normalizes ...))
    (file->value goldens-path))

  (define (shaped vs shape dt)
    (to-dtype (apply reshape (tensor vs #:dtype 'float64) shape) dt))

  (define (flat t)
    (~> t (to-dtype 'float64) (reshape -1) tensor->list))

  (define (source dt)
    (if (eq? dt 'uint8)
        (shaped byte-values byte-shape 'uint8)
        (shaped image-values image-shape dt)))

  (define inputs
    `((image ,image-values ,image-shape) (batch ,batch-values ,batch-shape)))

  (test-case "convert-image-dtype: every dtype pair, bit for bit torchvision 0.29's"
    (for ([golden (in-list converts)])
      (match-define (list from target want) golden)
      (define x (source from))
      (define got (convert-image-dtype x target))
      (define label (format "~a to ~a" from target))
      (check-equal? (dtype got) target label)
      (check-equal? (tensor-shape got) (tensor-shape x) label)
      (check-equal? (flat got) want label)
      (when (eq? from target)
        (check-eq? got x label))))

  (test-case "normalize: bit for bit torchvision 0.29's, on a copy or in place"
    (for ([golden (in-list normalizes)])
      (match-define (list dt input mean std inplace? want) golden)
      (match-define (list _ vs shape) (assq input inputs))
      (define x (shaped vs shape dt))
      (define before (flat x))
      (define got (normalize x mean std #:inplace? inplace?))
      (define label (format "~a ~a ~a ~a" dt input mean std))
      (check-equal? (dtype got) dt label)
      (check-equal? (tensor-shape got) shape label)
      (check-equal? (flat got) want label)
      (if inplace?
          (check-eq? got x label)
          (check-equal? (flat x) before label))))

  (define ((blamed-for? rx) e)
    (and (exn:fail:contract:blame? e)
         (regexp-match? rx (exn-message e))))

  (test-case "normalize: a list of another length or a zero std blames the caller"
    (define x (rand 2 3 4 4))
    (check-exn (blamed-for?
                #rx"^normalize: contract violation;\n one mean value per channel, or one for all")
               (lambda () (normalize x '(0.1 0.2) '(1 1 1))))
    (check-exn (blamed-for? #rx"one std value per channel, or one for all")
               (lambda () (normalize x '(0.1 0.2 0.3) '())))
    (check-exn (blamed-for?
                #rx"^normalize: contract violation;\n std evaluated to zero, leading to division by zero")
               (lambda () (normalize x '(0.1 0.2 0.3) '(1 0 1))))
    (check-exn (blamed-for? #rx"std evaluated to zero")
               (lambda () (normalize x '(0.1) '(-0.0))))
    (check-exn (blamed-for? #rx"std evaluated to zero")
               (lambda () (normalize x '(0.1) '(0)))))

  (define golden-op-calls
    '((convert-from-bytes 7) (normalize 14) (normalize-in-place 12)))

  (define (calls thunk)
    (thunk)
    (op-calls-of thunk))

  (test-case "convert from bytes is a cast and an in-place divide; normalize, two small tensors, a subtraction and an in-place divide"
    (define byte-image (source 'uint8))
    (define x (rand 3 8 8))
    (define counted
      `((convert-from-bytes ,(calls (lambda () (convert-image-dtype byte-image))))
        (normalize ,(calls (lambda () (normalize x imagenet-mean imagenet-std))))
        (normalize-in-place
         ,(calls (lambda () (normalize x imagenet-mean imagenet-std #:inplace? #t))))))
    (check-equal? counted golden-op-calls)))
