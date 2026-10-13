#lang racket/base

#|review: ignore|#

;; Re-export facade: raco review reads its whole-module requires as unused.
(require racket/require
         (subtract-in "main.rkt" "safe.rkt"))

(provide (all-from-out "main.rkt"))
