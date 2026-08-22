#lang racket/base

;; raco review lints without macro expansion and cannot see the re-export.
#|review: ignore|#

;; Facade for the high-level torch API.  `(require torch)` exposes the
;; ergonomic surface; for v0 it is exactly the safe contracted layer from
;; `foreign.rkt`.  A `tensor` is GC-reclaimed, so user code never frees it.

(require "foreign.rkt"
         (only-in threading ~> ~>> lambda~> lambda~>>))

(provide (all-from-out "foreign.rkt")
         ~> ~>> lambda~> lambda~>>)

(module+ main
  (printf "libtorch version: ~a\n" (torch-version))
  (manual-seed! 0)
  (define t (randn 2 2))
  (printf "randn ~a (PyTorch repr):\n" (tensor-shape t))
  (println t)
  (printf "ATen form:\n~a\n" (tensor->string t)))
