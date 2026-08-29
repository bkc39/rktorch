#lang racket/base

(require (only-in racket/contract
                  -> >=/c and/c define/contract flat-named-contract)
         (only-in racket/string non-empty-string? string-split)
         (only-in "functional.rkt" edit-distance))

(provide cer
         wer)

(define transcript/c
  (flat-named-contract
   'transcript-with-words
   (lambda (s) (and (string? s) (pair? (string-split s))))))

(define rate/c (and/c rational? (>=/c 0)))

(define/contract (wer reference hypothesis)
  (-> transcript/c string? rate/c)
  (define ref-words (string-split reference))
  (/ (edit-distance ref-words (string-split hypothesis))
     (length ref-words)))

(define/contract (cer reference hypothesis)
  (-> non-empty-string? string? rate/c)
  (/ (edit-distance (string->list reference) (string->list hypothesis))
     (string-length reference)))
