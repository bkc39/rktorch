#lang racket/base

;; torch.nn.functional (F): the functional forms of the conv / pool / flatten
;; ops whose bare names also name nn layer constructors (conv2d, max-pool2d,
;; flatten). They live here -- not on the `torch` top-level -- so that
;; `(require torch torch/nn)` no longer collides on those names (#11). Import as
;;
;;   (require (prefix-in F: torch/nn/functional))
;;   ... (F:max-pool2d x 2) (F:conv2d x w) (F:flatten h 1) ...
;;
;; mirroring PyTorch's `import torch.nn.functional as F`. avg-pool2d and
;; adaptive-avg-pool2d have no layer twin (no collision) so they also remain on
;; `torch`; they are re-exported here so F is the complete functional pooling
;; surface. These are the contracted bindings from foreign.rkt.

(require "../foreign.rkt")

(provide conv2d
         max-pool2d
         flatten
         avg-pool2d
         adaptive-avg-pool2d)
