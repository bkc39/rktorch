#lang racket/base

(provide main
         quit)

(define (main _ch)
  (sync never-evt))

(define (quit _ch)
  (void))
