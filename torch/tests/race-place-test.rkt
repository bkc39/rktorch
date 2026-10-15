#lang racket/base

(require (only-in ffi/unsafe _intptr _pointer cast cpointer?) ;; noqa
         (only-in racket/place place place-channel-get place-channel-put))

(define (echo-place)
  (place ch
    (define v (place-channel-get ch))
    (place-channel-put ch (list (cpointer? v) v))))

(module+ test
  (require (only-in racket/future future touch)
           (only-in racket/match match-define)
           (only-in racket/place place-wait)
           rackunit
           (only-in "../foreign.rkt" shape tensor? zeros))

  (test-case "a tensor op inside a future runs when the future is touched"
    (define v (touch (future (lambda () (zeros 2 3)))))
    (check-true (tensor? v))
    (check-equal? (shape v) '(2 3)))

  (test-case "a tensor sent to another place arrives as a bare pointer"
    (define t (zeros 2 2))
    (define p (echo-place))
    (place-channel-put p t)
    (match-define (list pointer-there? echoed) (place-channel-get p))
    (check-equal? (place-wait p) 0)
    (check-true pointer-there?)
    (check-false (tensor? echoed))
    (check-equal? (cast echoed _pointer _intptr) (cast t _pointer _intptr)
                  "the message points at the sender's storage")))
