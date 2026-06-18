#lang racket/base

;; raco review does surface-level linting without macro expansion, so it cannot
;; see the re-export below; mark this pure facade as ignore.
#|review: ignore|#

;; Facade for the high-level torch API.  `(require torch)` exposes the
;; ergonomic surface; for v0 it is exactly the safe contracted layer from
;; `foreign.rkt`.  A `tensor` is GC-reclaimed, so user code never frees it.

(require "foreign.rkt"
         (only-in threading ~> ~>> lambda~> lambda~>>))

;; Re-provide the threading library's pipeline operators so `(require torch)`
;; yields `~>` for tensor pipelines (mirrors rkt-polars):
;;   (~> x (* x) Σ)  ==  (Σ (* x x))
;;
;; conv2d / max-pool2d / flatten are withheld from the top-level surface: their
;; bare names are nn layer constructors (torch/nn), so exporting the functional
;; forms here too would collide under `(require torch torch/nn)` (#11). The
;; functional forms live in torch/nn/functional (F). avg-pool2d /
;; adaptive-avg-pool2d have no layer twin, so they stay here.
(provide (except-out (all-from-out "foreign.rkt")
                     conv2d max-pool2d flatten)
         ~> ~>> lambda~> lambda~>>)

(module+ main
  (printf "libtorch version: ~a\n" (torch-version))
  (manual-seed! 0)
  (define t (randn 2 2))
  (printf "randn ~a (PyTorch repr):\n" (tensor-shape t))
  (println t)
  (printf "ATen form:\n~a\n" (tensor->string t)))
