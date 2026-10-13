#lang racket/base

#|review: ignore|#

(require racket/require
         (subtract-in "main.rkt" "safe.rkt"))

(provide (all-from-out "main.rkt"))
