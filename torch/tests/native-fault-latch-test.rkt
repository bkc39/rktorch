#lang racket/base

;; This file latches its own process on purpose, so it holds one test case
;; whose steps run in order; nothing native can run after the latch.
(module+ test
  (require (only-in ffi/unsafe _intptr cast)
           rackunit
           (only-in "../foreign.rkt" ones)
           (only-in "../foreign/raw/fault.rkt" latched native-faulted)
           (only-in "../foreign/raw/syntax.rkt" _Tensor)
           (only-in "../foreign/structs.rkt" tensor-impl))

  (define arities
    (list (latched (lambda () (list)))
          (latched (lambda (a) (list a)))
          (latched (lambda (a b) (list a b)))
          (latched (lambda (a b c) (list a b c)))
          (latched (lambda (a b c d) (list a b c d)))
          (latched (lambda (a b c d e) (list a b c d e)))
          (latched (lambda (a b c d e f) (list a b c d e f)))
          (latched (lambda (a b c d e f g) (list a b c d e f g)))))

  (define (call-each)
    (for/list ([p (in-list arities)] [n (in-naturals)])
      (apply p (build-list n values))))

  (define disabled #rx"^rktorch: native code faulted earlier, while printing a tensor")

  (test-case "the first fault in the printer disables every native call after it"
    (check-equal? (call-each) (for/list ([n (in-range 8)]) (build-list n values))
                  "before any fault the latch passes every argument through")
    (check-false (native-faulted))
    (check-equal? (format "~a" (tensor-impl (cast 8 _intptr _Tensor) '(2 2)))
                  "#<tensor:2x2>"
                  "the printer falls back to the shape")
    (check-equal? (native-faulted) "printing a tensor")
    (for ([p (in-list arities)] [n (in-naturals)])
      (check-exn disabled (lambda () (apply p (build-list n values)))))
    (check-exn disabled (lambda () (ones 2 2))
               "a real binding raises without entering native code")))
