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
;; The functional ops keep their lowercase names here (conv2d / max-pool2d /
;; flatten / …), mirroring `torch.conv2d`. The nn layer constructors are
;; PascalCase (Conv2d / MaxPool2d / Flatten / Linear / …, mirroring the
;; `torch.nn.*` classes), so `(require torch torch/nn)` never collides (#11).
(provide (all-from-out "foreign.rkt")
         ~> ~>> lambda~> lambda~>>)

(module+ main
  (printf "libtorch version: ~a\n" (torch-version))
  (manual-seed! 0)
  (define t (randn 2 2))
  (printf "randn ~a (PyTorch repr):\n" (tensor-shape t))
  (println t)
  (printf "ATen form:\n~a\n" (tensor->string t)))
