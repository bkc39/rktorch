#lang racket/base

#|review: ignore|#

(require racket/require
         (subtract-in "main.rkt" racket))

(provide (all-from-out "main.rkt"))
