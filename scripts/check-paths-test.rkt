#lang racket/base

(require (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in rackunit check-equal? test-case)
         (only-in "check-paths.rkt" check-paths mention stale-mentions))

(define-runtime-path scripts ".")

(define doc "check-paths-fixture/doc.txt")
(define note "check-paths-fixture/inner/note.txt")

(define stale
  (list (mention doc 3 "gone.rkt")
        (mention doc 4 "inner/gone.txt")
        (mention note 2 "test/gone.py")))

(test-case "a path resolves from the root or from its file's directory"
  (check-equal? (stale-mentions scripts (list doc note)) stale))

(test-case "an allowance hides its mention, and one that hides nothing is reported"
  (define-values (left unused)
    (check-paths scripts (list doc note)
                 (list (list doc "gone.rkt") (list doc "inner/note.txt"))))
  (check-equal? left (cdr stale))
  (check-equal? unused (list (list doc "inner/note.txt"))))
