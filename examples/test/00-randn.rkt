#lang racket/base

;; Runner + tests for the literate example ../../examples/racket/00-randn.rkt.
;; The example itself is a #lang scribble/lp2 program (woven into the manual);
;; lp2 submodules can't see chunk-level bindings, so main/test live here and
;; require the example's provides.

(require torchrkt
         "../racket/00-randn.rkt")

(module+ main
  (define t (run-example))
  (printf "shape: ~a\n" (tensor-shape t))
  (display (tensor->string t))
  (newline))

(module+ test
  (require rackunit)
  (define t (run-example))
  (check-equal? (tensor-shape t) '(2 2))
  (check-equal? (tensor-numel t) 4)
  (check-equal? (length (tensor->list t)) 4))
