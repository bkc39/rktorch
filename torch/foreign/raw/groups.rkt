#lang racket/base

(require (only-in "cas.rkt" cas-add! cas-claim!)
         (only-in "race-points.rkt" race-point))

(provide fresh-group
         group-add!
         group-used?
         lower-group!
         settle!)

(module+ private
  (provide group-shown))

(struct group (count [phantom #:mutable] [shown #:mutable]))

;; `count` packs a group's bytes above `request-bits` and its requests, the
;; adds since its last settle, below them. The add that takes the requests
;; from 0 to 1 makes its thread the group's one settler; a fresh group starts
;; at 1, held by its creator. At most 64 entries and two adds an entry, so
;; the requests never carry into the bytes. A charge adds before it publishes
;; its entry, so the bytes are never negative.
(define request-bits 12)
(define request-mask (sub1 (arithmetic-shift 1 request-bits)))

(define (requests-of count) (bitwise-and count request-mask))
(define (bytes-of count) (arithmetic-shift count (- request-bits)))

(define (fresh-group nbytes)
  (group (box (add1 (arithmetic-shift nbytes request-bits))) #f 0))

(define (group-used? g)
  (< 1 (requests-of (unbox (group-count g)))))

(define (group-add! g nbytes)
  (define count (cas-add! (group-count g) (add1 (arithmetic-shift nbytes request-bits))
                          #:site 'group))
  (eqv? 1 (requests-of count)))

(define (show! g target)
  (define was (group-shown g))
  (unless (eqv? was target)
    (define phantom (group-phantom g))
    (cond
      [(not phantom) (set-group-phantom! g (make-phantom-bytes target))]
      [else (set-phantom-bytes! phantom target)
            (when (zero? target)
              (set-group-phantom! g #f))])
    (set-group-shown! g target)
    (race-point group-shown (cons g (- target was)))))

(define (settle! g)
  (define count (group-count g))
  (let retry ()
    (define seen (unbox count))
    (show! g (bytes-of seen))
    (unless (cas-claim! count seen (- seen (requests-of seen)) #:site 'group)
      (retry))))

(define (lower-group! g nbytes)
  (when (group-add! g (- nbytes))
    (settle! g)))
