#lang racket/base
(require torch)

(define (try thunk)
  (with-handlers ([exn:fail? (lambda (e)
                               (format "RAISED: ~a"
                                       (car (regexp-split #rx"\n"
                                                          (exn-message e)))))])
    (thunk)))

(printf "full 0.5 int64      -> ~a\n"
        (try (lambda () (tensor->list (full 0.5 2 #:dtype 'int64)))))
(printf "full 2.0 int64      -> ~a\n"
        (try (lambda () (tensor->list (full 2.0 2 #:dtype 'int64)))))
(printf "full 1/2 int64      -> ~a\n"
        (try (lambda () (tensor->list (full 1/2 2 #:dtype 'int64)))))
(printf "full 0.5 uint8      -> ~a\n"
        (try (lambda () (tensor->list (full 0.5 2 #:dtype 'uint8)))))
