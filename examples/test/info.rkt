#lang info

;; raco review lints this as a normal module and flags every `info` definition
;; as unused; #lang info has no value-level uses to detect.
#|review: ignore|#

;; Test harnesses for the literate `examples/racket/*.rkt` programs.  Each
;; `NN-name.rkt` here requires its `#lang scribble/lp2` sibling, drives it from
;; `(module+ main ...)`, and checks it from `(module+ test ...)`.  Run the suite
;; with `raco test examples/test/`.

(define test-timeouts
  '(("00-randn.rkt" 300)))
