#lang racket/base

(module+ test
  (require rackunit
           "../data/loader.rkt"
           (prefix-in g: (only-in "../generated.rkt" multinomial))
           "../main.rkt"
           (only-in "../vision/transforms.rkt"
                    random-crop random-horizontal-flip random-resized-crop)
           (only-in "private/blame.rkt" blamed contract-violation?))

  (define caller (blamed "current-generator-test.rkt"))

  (define (drawn v)
    (if (tensor? v) (tensor->list v) v))

  (define images (~> (* 4 3 6 6) arange (reshape 4 3 6 6)))
  (define probabilities (tensor '(0.1 0.2 0.3 0.4 0.5)))

  (define random-ops
    (list (cons 'rand (lambda (g) (rand 2 3 #:generator g)))
          (cons 'randn (lambda (g) (randn 2 3 #:generator g)))
          (cons 'rand-like (lambda (g) (rand-like (zeros 4) #:generator g)))
          (cons 'randn-like (lambda (g) (randn-like (zeros 4) #:generator g)))
          (cons 'randperm (lambda (g) (randperm 8 #:generator g)))
          (cons 'draw-seed (lambda (g) (draw-seed #:generator g)))
          (cons 'multinomial
                (lambda (g) (multinomial probabilities 3 #:generator g)))))

  (define transforms
    (list (cons 'random-horizontal-flip
                (lambda (g) (random-horizontal-flip images #:generator g)))
          (cons 'random-crop
                (lambda (g) (random-crop images #:padding 2 #:generator g)))
          (cons 'random-resized-crop
                (lambda (g) (random-resized-crop images 4 #:generator g)))))

  (define (check-resolves ops)
    (for ([op (in-list ops)])
      (define via-parameter
        (parameterize ([current-generator (make-generator 13)])
          ((cdr op) #f)))
      (check-equal? (drawn via-parameter)
                    (drawn ((cdr op) (make-generator 13)))
                    (format "~a through current-generator" (car op)))))

  (test-case "current-generator defaults to the global stream"
    (check-false (current-generator))
    (manual-seed! 21)
    (define seed (draw-seed))
    (define perm (tensor->list (randperm 8)))
    (define twin (make-generator 21))
    (check-equal? seed (draw-seed #:generator twin)
                  "manual-seed! s seeds the stream a fresh generator s draws")
    (check-equal? perm (tensor->list (randperm 8 #:generator twin))))

  (test-case "random ops given #f draw from current-generator"
    (check-resolves random-ops))

  (test-case "a generated op's optional generator resolves through current-generator"
    (define via-parameter
      (parameterize ([current-generator (make-generator 13)])
        (g:multinomial probabilities 3 #f #f)))
    (check-equal? (tensor->list via-parameter)
                  (tensor->list (g:multinomial probabilities 3 #f
                                               (make-generator 13)))))

  (test-case "random transforms seed from current-generator"
    (check-resolves transforms))

  (test-case "an explicit #:generator wins over current-generator"
    (define g1 (make-generator 1))
    (define explicit
      (parameterize ([current-generator g1])
        (tensor->list (randperm 8 #:generator (make-generator 2)))))
    (check-equal? explicit (tensor->list (randperm 8 #:generator (make-generator 2))))
    (check-equal? (tensor->list (randperm 8 #:generator g1))
                  (tensor->list (randperm 8 #:generator (make-generator 1)))
                  "g1 drew nothing while the explicit generator was in use"))

  (test-case "parameterizing current-generator leaves the global stream alone"
    (manual-seed! 5)
    (define baseline (tensor->list (randn 4)))
    (manual-seed! 5)
    (parameterize ([current-generator (make-generator 8)])
      (void (randn 4) (randperm 6) (draw-seed)))
    (check-equal? (tensor->list (randn 4)) baseline))

  (test-case "current-generator rejects a non-generator, blaming the caller"
    (check-exn (contract-violation? caller #rx"generator")
               (lambda () (parameterize ([current-generator 5]) (randperm 3))))
    (check-exn (contract-violation? caller #rx"generator")
               (lambda () (current-generator 'x))))

  (test-case "a loader without a generator draws from current-generator"
    (define loader
      (~> (arange 10 #:dtype 'int64)
          tensor-dataset
          (dataloader #:batch-size 3 #:shuffle? #t)))
    (define (order)
      (parameterize ([current-generator (make-generator 3)])
        (for/list ([ib loader]) (tensor->list ib))))
    (manual-seed! 5)
    (define baseline (tensor->list (randn 4)))
    (manual-seed! 5)
    (define first-order (order))
    (check-equal? (order) first-order)
    (check-equal? (tensor->list (randn 4)) baseline))

  (test-case "parameterizing current-generator is not passing it as #:generator"
    (define (shuffled g)
      (~> (arange 10 #:dtype 'int64)
          tensor-dataset
          (dataloader #:batch-size 10 #:shuffle? #t #:generator g)))
    (define (first-batch loader)
      (for/first ([b loader]) (tensor->list b)))
    (define passed (~> 3 make-generator shuffled first-batch))
    (define parameterized
      (parameterize ([current-generator (make-generator 3)])
        (first-batch (shuffled #f))))
    (define g (make-generator 3))
    (draw-seed #:generator g)
    (check-equal? passed (tensor->list (randperm 10 #:generator g)))
    (define h (make-generator 3))
    (draw-seed #:generator h)
    (check-equal? parameterized
                  (~> (draw-seed #:generator h)
                      make-generator
                      (randperm 10 #:generator _)
                      tensor->list))
    (check-not-equal? parameterized passed)))
