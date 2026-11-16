#lang racket/base

(require (only-in racket/contract/base ->i)
         (only-in racket/match match-define)
         (only-in "../../private/contract.rkt" define/contract-out)
         (only-in "../mnist.rkt" read-idx))

(define (record-count bs)
  (define-values (dims _data) (read-idx bs))
  (match-define (cons n _) dims)
  n)

(define/contract-out (idx-head bs n)
  (->i ([bs bytes?] [n exact-nonnegative-integer?])
       #:pre/name (bs n) "n exceeds the buffer's record count"
       (<= n (record-count bs))
       [_ bytes?])
  (define ndim (bytes-ref bs 3))
  (define-values (dims _data) (read-idx bs))
  (match-define (cons _ record-shape) dims)
  (define record (apply * record-shape))
  (define start (+ 4 (* 4 ndim)))
  (bytes-append (subbytes bs 0 4)
                (integer->integer-bytes n 4 #f #t)
                (subbytes bs 8 start)
                (subbytes bs start (+ start (* n record)))))
