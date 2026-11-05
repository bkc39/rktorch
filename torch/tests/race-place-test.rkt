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
           (only-in "../foreign.rkt" shape tensor? zeros)
           (only-in "private/wedge.rkt" exit-status-of))

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
                  "the message points at the sender's storage"))

  (define place-exit-program
    '(begin
       (require (only-in racket/place dynamic-place place-dead-evt)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (call-with-process-deadline
        50
        (lambda ()
          (define p (dynamic-place (collection-file-path "place-reapers.rkt" "torch" "tests" "private")
                                   'main))
          (define churned (sync/timeout 30 p))
          (define dead (sync/timeout 10 (place-dead-evt p)))
          (eprintf "place ~a, dead within 10 s of returning: ~a\n" churned (and dead #t))
          (exit (if (and churned dead) 0 3))))))

  (test-case "L14: a place whose 'own workers made tensors dies when its body returns"
    (check-equal? (exit-status-of place-exit-program #:within 90) 0)))
