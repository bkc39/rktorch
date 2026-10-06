#lang racket/base

;; raco cover runs every test file in one process, so the latch this file
;; sets is cleared before the next one runs.
(module+ test
  (require (only-in ffi/unsafe _intptr cast)
           rackunit
           (only-in "../foreign.rkt" ones tensor-shape)
           (only-in "../foreign/raw/fault.rkt"
                    latched native-faulted reset-native-fault-latch!)
           (only-in "../foreign/raw/memory.rkt" reaccount!)
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

  (define (bad) (cast 8 _intptr _Tensor))

  (define (disabled doing)
    (regexp (format "^rktorch: native code faulted earlier, while ~a" doing)))

  (define passed-through (for/list ([n (in-range 8)]) (build-list n values)))

  (test-case "the first fault in the printer disables every native call after it"
    (dynamic-wind
     void
     (lambda ()
       (check-equal? (call-each) passed-through
                     "before any fault the latch passes every argument through")
       (check-false (native-faulted))
       (check-equal? (format "~a" (tensor-impl (bad) '(2 2)))
                     "#<tensor:2x2>"
                     "the printer falls back to the shape")
       (check-equal? (native-faulted) "printing a tensor")
       (for ([p (in-list arities)] [n (in-naturals)])
         (check-exn (disabled "printing a tensor")
                    (lambda () (apply p (build-list n values)))))
       (check-exn (disabled "printing a tensor") (lambda () (ones 2 2))
                  "a real binding raises without entering native code"))
     reset-native-fault-latch!))

  (test-case "a fault while accounting a tensor disables the library too"
    (dynamic-wind
     void
     (lambda ()
       (check-false (native-faulted))
       (check-not-exn (lambda () (reaccount! (bad)))
                      "the accounting swallows the fault")
       (check-equal? (native-faulted) "accounting a tensor")
       (check-exn (disabled "accounting a tensor") (lambda () (ones 2 2))))
     reset-native-fault-latch!))

  (test-case "clearing the latch lets native calls through again"
    (check-false (native-faulted))
    (check-equal? (call-each) passed-through)
    (check-equal? (tensor-shape (ones 2 2)) '(2 2))))
