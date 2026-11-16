#lang racket/base

(require (only-in racket/contract/base ->i flat-named-contract)
         (only-in racket/match match-define)
         (only-in "../../private/contract.rkt" define/contract-out))

(define (header-end bs)
  (and (>= (bytes-length bs) 4)
       (bytes=? (subbytes bs 0 3) #"\0\0\10")
       (positive? (bytes-ref bs 3))
       (+ 4 (* 4 (bytes-ref bs 3)))))

(define (layout bs)
  (define end (header-end bs))
  (cond
    [(not end) #f]
    [(< (bytes-length bs) end) #f]
    [else
     (match-define (cons n record-shape)
       (for/list ([at (in-range 4 end 4)])
         (integer-bytes->integer bs #f #t at (+ at 4))))
     (list n (apply * record-shape) end)]))

(define (holds-records? bs n)
  (match-define (list count record start) (layout bs))
  (and (<= n count) (<= (+ start (* n record)) (bytes-length bs))))

(define idx-bytes/c
  (flat-named-contract "a uint8 IDX buffer" (lambda (bs) (and (bytes? bs) (layout bs) #t))))

(define/contract-out (idx-head bs n)
  (->i ([bs idx-bytes/c] [n exact-nonnegative-integer?])
       #:pre/name (bs n) "n exceeds the buffer's records"
       (holds-records? bs n)
       [_ bytes?])
  (match-define (list _ record start) (layout bs))
  (bytes-append (subbytes bs 0 4)
                (integer->integer-bytes n 4 #f #t)
                (subbytes bs 8 start)
                (subbytes bs start (+ start (* n record)))))
