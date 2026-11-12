#lang racket/base

(require (only-in racket/match match match-define match-lambda struct*)
         (only-in "cas.rkt" cas-claim! cas-update! with-claim)
         (only-in "race-points.rkt" race-point))

(provide fresh-phantoms
         charge-epoch
         young-epoch?
         no-young
         gather-young
         flushed-young
         note-young!
         sync-phantoms!)

(module+ private
  (provide phantom-bytes rounds young-count-of))

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

(struct young-count (epoch bytes))

(define no-young (young-count (make-weak-box #f) 0))

(define (gather-young held nbytes)
  (match held
    [(young-count (? young-epoch? e) n) (young-count e (+ n nbytes))]
    [(young-count _ n) (young-count (charge-epoch) (+ (max 0 n) nbytes))]))

(define flushed-young
  (match-lambda
    [(young-count (? young-epoch?) n) n]
    [(young-count _ n) (max 0 n)]))

(struct phantoms
  (claim count old
   [young #:mutable] [epoch #:mutable]
   [old-bytes #:mutable] [young-bytes #:mutable]))

(define (fresh-phantoms)
  (define e (charge-epoch))
  (phantoms (box #f) (box (young-count e 0)) (make-phantom-bytes 0) (make-phantom-bytes 0) e 0 0))

(define (note-young! ph nbytes)
  (define e (charge-epoch))
  (define (current? held) (or (eq? held e) (young-epoch? held)))
  (unless (eqv? 0 nbytes)
    (cas-update! (phantoms-count ph)
                 (match-lambda
                   [(young-count (? current? held) n) (young-count held (max 0 (+ n nbytes)))]
                   [_ (young-count e (max 0 nbytes))])
                 #:site 'phantom)))

(define (phantom-bytes ph)
  (values (phantoms-old-bytes ph) (phantoms-young-bytes ph)))

(define (young-count-of ph)
  (young-count-bytes (unbox (phantoms-count ph))))

(define (targets ph live)
  (match-define (struct* phantoms ([count count])) ph)
  (match-define (young-count e n) (unbox count))
  (define l (max 0 (unbox live)))
  (define y (if (young-epoch? e) (max 0 (min l n)) 0))
  (values e (- l y) y))

(define (roll! ph e)
  (set-young! ph 0)
  (set-phantoms-young! ph (make-phantom-bytes 0))
  (set-phantoms-epoch! ph e))

(define (set-old! ph n)
  (match-define (struct* phantoms ([old old] [old-bytes was])) ph)
  (unless (eqv? n was)
    (set-phantom-bytes! old n)
    (set-phantoms-old-bytes! ph n)))

(define (set-young! ph n)
  (match-define (struct* phantoms ([young young] [young-bytes was])) ph)
  (unless (eqv? n was)
    (set-phantom-bytes! young n)
    (set-phantoms-young-bytes! ph n)))

(define (apply-targets! ph e old young)
  (match-define (struct* phantoms ([epoch held] [young-bytes was])) ph)
  (define rolls? (not (or (eqv? 0 young) (eq? e held))))
  (define lower-first? (< young was))
  (cond
    [rolls? (roll! ph e) (set-old! ph old) (set-young! ph young)]
    [lower-first? (set-young! ph young) (set-old! ph old)]
    [else (set-old! ph old) (set-young! ph young)]))

(define (applied? ph live)
  (define-values (e old young) (targets ph live))
  (match-define (struct* phantoms ([epoch held] [old-bytes old-was] [young-bytes young-was])) ph)
  (and (eqv? old old-was)
       (eqv? young young-was)
       (or (eqv? 0 young) (eq? e held))))

(define rounds 4)

(define (sync-round! ph live)
  (with-claim (phantoms-claim ph) #:site 'phantom
    (define-values (e old young) (targets ph live))
    (apply-targets! ph e old young)
    (race-point phantoms-applied ph)
    #t))

(define (settled? ph live)
  (memory-order-release)
  (and (not (unbox (phantoms-claim ph)))
       (begin
         (memory-order-acquire)
         (applied? ph live))))

(define (sync-phantoms! ph live)
  (unless (settled? ph live)
    (for/and ([_ (in-range rounds)])
      (and (sync-round! ph live)
           (not (applied? ph live))))
    (void)))
