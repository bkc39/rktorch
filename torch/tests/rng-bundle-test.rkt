#lang racket/base

(module+ test
  (require (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../main.rkt" current-generator make-generator rand-like randn tensor->list zeros)
           "../train.rkt"
           (only-in "../train/private/value.rkt" checkpoint-value?))

  (define-runtime-path restore-module "../train/private/restore.rkt")

  (define (draws)
    (list (tensor->list (randn 4))
          (tensor->list (rand-like (zeros 3)))
          (for/list ([_ (in-range 3)]) (random))))

  (test-case "T3-L15: capture, draw, restore, draw: the same draws"
    (define rng-bundle (dynamic-require restore-module 'rng-bundle))
    (define restore-rng-bundle! (dynamic-require restore-module 'restore-rng-bundle!))
    (parameterize ([current-generator (make-generator 21)])
      (randn 2)
      (random)
      (define bundle (rng-bundle))
      (check-true (checkpoint-value? bundle) "the bundle is a checkpoint value")
      (define first-draws (draws))
      (restore-rng-bundle! bundle)
      (check-equal? (draws) first-draws))))
