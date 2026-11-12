#lang racket/base

(require (only-in racket/match match-define match-lambda)
         (only-in "cas.rkt" cas-claim! cas-update!)
         (only-in "race-points.rkt" race-point))

(provide fresh-phantoms
         charge-epoch
         young-epoch?
         note-young!
         sync-phantoms!)

(module+ private
  (provide phantom-bytes))

;; A collection clears the sentinel, which nothing else holds.
(define (fresh-epoch)
  (make-weak-box (box 0)))

(define epoch (box (fresh-epoch)))

(define (charge-epoch)
  (define e (unbox epoch))
  (cond
    [(weak-box-value e) e]
    [else
     (cas-claim! epoch e (fresh-epoch) #:site 'phantom)
     (unbox epoch)]))

(define (young-epoch? e)
  (and (weak-box-value e) #t))

;; `young-count` is the bytes charged in its epoch, net of that epoch's
;; releases, as flushed and never below 0: a pair of the epoch and the count,
;; so the first count after a collection starts a fresh one.
(struct phantoms
  (claim young-count old
   [young #:mutable] [epoch #:mutable]
   [old-bytes #:mutable] [young-bytes #:mutable]))

(define (fresh-phantoms)
  (define e (charge-epoch))
  (phantoms (box #f) (box (cons e 0)) (make-phantom-bytes 0) (make-phantom-bytes 0) e 0 0))

(define (note-young! ph nbytes)
  (unless (eqv? 0 nbytes)
    (define e (charge-epoch))
    (cas-update! (phantoms-young-count ph)
                 (match-lambda
                   [(cons held n)
                    (if (or (eq? held e) (young-epoch? held))
                        (cons held (max 0 (+ n nbytes)))
                        (cons e (max 0 nbytes)))])
                 #:site 'phantom)))

(define (phantom-bytes ph)
  (values (phantoms-old-bytes ph) (phantoms-young-bytes ph)))

(define (targets ph live)
  (define l (max 0 (unbox live)))
  (match-define (cons e n) (unbox (phantoms-young-count ph)))
  (define y (if (young-epoch? e) (min l n) 0))
  (values e (- l y) y))

(define (roll! ph e)
  (set-young! ph 0)
  (set-phantoms-young! ph (make-phantom-bytes 0))
  (set-phantoms-epoch! ph e))

(define (set-old! ph n)
  (unless (eqv? n (phantoms-old-bytes ph))
    (set-phantom-bytes! (phantoms-old ph) n)
    (set-phantoms-old-bytes! ph n)))

(define (set-young! ph n)
  (unless (eqv? n (phantoms-young-bytes ph))
    (set-phantom-bytes! (phantoms-young ph) n)
    (set-phantoms-young-bytes! ph n)))

(define (apply-targets! ph e old young)
  (unless (or (eqv? 0 young) (eq? e (phantoms-epoch ph)))
    (roll! ph e))
  (cond
    [(< young (phantoms-young-bytes ph)) (set-young! ph young) (set-old! ph old)]
    [else (set-old! ph old) (set-young! ph young)]))

(define (applied? ph live)
  (define-values (e old young) (targets ph live))
  (and (eqv? old (phantoms-old-bytes ph))
       (eqv? young (phantoms-young-bytes ph))
       (or (eqv? 0 young) (eq? e (phantoms-epoch ph)))))

;; A thread that finds the claim held leaves the work to its holder, which
;; looks again after letting go; the letting go is a CAS so that look cannot
;; pass it. The holder looks again a bounded number of times, so threads that
;; keep flushing cannot hold it there; what it leaves goes with the next sync,
;; and every reader's `flush-tallies!` syncs. The caller is uninterruptible, so
;; a kill never strands the claim.
(define rounds 4)

(define (sync-phantoms! ph live)
  (define claim (phantoms-claim ph))
  (when (cas-claim! claim #f #t #:site 'phantom)
    (let again ([round 1])
      (define-values (e old young) (targets ph live))
      (apply-targets! ph e old young)
      (race-point phantoms-applied ph)
      (cas-claim! claim #t #f #:site 'phantom)
      (when (and (< round rounds)
                 (not (applied? ph live))
                 (cas-claim! claim #f #t #:site 'phantom))
        (again (add1 round))))))
