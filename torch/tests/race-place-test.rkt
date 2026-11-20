#lang racket/base

(require (only-in ffi/unsafe _intptr _pointer cast cpointer?) ;; noqa
         (only-in racket/place place place-channel-get place-channel-put))

(define (echo-place)
  (place ch
    (define v (place-channel-get ch))
    (place-channel-put ch (list (cpointer? v) v))))

(module+ test
  (require (only-in racket/format ~a)
           (only-in racket/future future touch)
           (only-in racket/match match match-define)
           (only-in racket/place dynamic-place place-dead-evt place-kill place-wait)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../foreign.rkt" shape tensor? zeros)
           (only-in "private/wedge.rkt" exit-status-of))

  (test-case "a tensor op inside a future runs when the future is touched"
    (define v (touch (future (lambda () (zeros 2 3)))))
    (check-true (tensor? v))
    (check-equal? (shape v) '(2 3)))

  (define (reply-within p seconds)
    (sync/timeout seconds p (handle-evt (place-dead-evt p) (lambda (_) (sync/timeout 0 p)))))

  (define-runtime-path place-stall "private/place-stall.rkt")

  (test-case "a place that ends or stalls without answering is not waited on past its bound"
    (for ([entry (in-list '(quit main))])
      (define p (dynamic-place place-stall entry))
      (define start (current-inexact-monotonic-milliseconds))
      (check-false (reply-within p (if (eq? entry 'quit) 60 1)) (~a entry))
      (check-true (< (- (current-inexact-monotonic-milliseconds) start) 30000) (~a entry))
      (place-kill p)))

  (test-case "a tensor sent to another place arrives as a bare pointer"
    (define t (zeros 2 2))
    (define p (echo-place))
    (place-channel-put p t)
    (define reply (reply-within p 120))
    (define status (and (sync/timeout 30 (place-dead-evt p)) (place-wait p)))
    (place-kill p)
    (match-define (list pointer-there? echoed)
      (match reply
        [(list _ _) reply]
        [_ (fail "the echo place died or did not answer within 120 s")]))
    (check-equal? status 0)
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
    (check-equal? (exit-status-of place-exit-program #:within 90) 0))

  (define place-exits 100)

  (define place-exits-program
    `(begin
       (require (only-in racket/place dynamic-place place-dead-evt)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (call-with-process-deadline
        280
        (lambda ()
          (define module (collection-file-path "place-exit.rkt" "torch" "tests" "private"))
          (define exited
            (for/sum ([_ (in-range ,place-exits)])
              (define p (dynamic-place module 'main))
              (define made (sync/timeout 30 p))
              (define dead (sync/timeout 10 (place-dead-evt p)))
              (if (and made dead) 1 0)))
          (eprintf "~a of ~a places made tensors and exited\n" exited ,place-exits)
          (exit (if (= exited ,place-exits) 0 3))))))

  (test-case "a hundred places that load torch and make tensors exit without crashing the process"
    (check-equal? (exit-status-of place-exits-program #:within 300) 0)))
