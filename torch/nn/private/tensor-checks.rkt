#lang racket/base

(require (only-in racket/match match-define)
         (only-in "../../foreign.rkt"
                  device dtype eq max min numel shape stack tensor->list tensor? where))

(provide in-range?
         rank-of
         rows
         same-device?
         same-rows?
         tensor-of?)

(define (rank-of v)
  (length (shape v)))

(define (tensor-of? v ranks dtype?)
  (and (tensor? v) (memv (rank-of v) ranks) (dtype? (dtype v)) #t))

(define (rows v)
  (match-define (cons n _) (shape v))
  n)

(define (same-rows? a b)
  (= (rows a) (rows b)))

(define (same-device? a b)
  (equal? (device a) (device b)))

(define (in-range? t bound [ignore-index #f])
  (or (zero? (numel t))
      (let ([kept (if ignore-index (where (eq t ignore-index) 0 t) t)])
        (match-define (list lo hi) (tensor->list (stack (list (min kept) (max kept)))))
        (and (<= 0 lo) (< hi bound)))))
