#lang racket/base

(require (only-in racket/match match))

(provide host-platform
         kernel-platform
         path-convention
         simulated-system-type)

(define (kernel-platform system-type)
  (match (system-type 'os*)
    [(or 'macosx 'darwin) 'macosx]
    [os* os*]))

(define (path-convention system-type)
  (system-type 'os))

(define host-platform (kernel-platform system-type))

(define ((simulated-system-type os os*) mode)
  (match mode
    ['os os]
    ['os* os*]))

(module+ test
  (require rackunit)

  (define racket-builds
    '((unix darwin macosx unix)
      (macosx macosx macosx macosx)
      (unix linux linux unix)
      (windows windows windows windows)))

  (for ([b (in-list racket-builds)])
    (match b
      [(list os os* kernel paths)
       (test-case (format "a Racket reporting os ~a and os* ~a runs on ~a with ~a paths"
                          os os* kernel paths)
         (check-equal? (kernel-platform (simulated-system-type os os*)) kernel)
         (check-equal? (path-convention (simulated-system-type os os*)) paths))])))
