#lang racket/base

(require (only-in "memory.rkt" define-binary/raw))

(provide tr-matmul/raw
         tr-mm/raw
         tr-mv/raw
         tr-dot/raw)

(define-binary/raw tr-matmul/raw tr_matmul)
(define-binary/raw tr-mm/raw tr_mm)
(define-binary/raw tr-mv/raw tr_mv)
(define-binary/raw tr-dot/raw tr_dot)
