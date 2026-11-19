#lang racket/base

(require (only-in racket/contract/base
                  ->* >=/c and/c flat-named-contract listof not/c)
         (only-in racket/list first group-by)
         (only-in racket/match == match)
         (only-in racket/math nan?)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  + - / cat clamp copy! device dtype item stack tensor? to-device zeros)
         (only-in "../foreign/autograd-ops.rkt" with-gradients)
         (only-in "../foreign/ops.rkt" move-generation)
         (prefix-in g: (only-in "../generated.rkt"
                                foreach-clamp-max-scalar! foreach-clamp-min-scalar!
                                foreach-mul-tensor! foreach-norm-scalar
                                linalg-vector-norm mul-tensor!))
         (only-in "../private/contract.rkt" define/contract-out))

(define norm-type/c (flat-named-contract 'norm-type/c (and/c real? (not/c nan?))))

(define/contract-out (clip-grad-norm! params max-norm ;; noqa
                                      #:norm-type [norm-type 2.0]
                                      #:error-if-nonfinite? [error-if-nonfinite? #f]
                                      #:foreach? [foreach? #t])
  (->* ((listof tensor?) (and/c real? (>=/c 0)))
       (#:norm-type norm-type/c #:error-if-nonfinite? boolean? #:foreach? boolean?)
       tensor?)
  (define ord (exact->inexact norm-type))
  (define kernels (kernels-for foreach?))
  (with-gradients ([qs gs] params)
    (match (buckets-of qs gs)
      ['() (zeros)]
      [buckets
       (define home (bucket-device (first buckets)))
       (define total (total-norm buckets home ord kernels))
       (when (and error-if-nonfinite? (~> total item rational? not))
         (error 'clip-grad-norm!
                (string-append "the total norm of order ~a of the gradients is non-finite, "
                               "so it cannot be clipped")
                ord))
       (define scale (clamp (/ max-norm (+ total 1e-6)) #:max 1.0))
       (for ([b (in-list buckets)])
         ((kernels-scale! kernels) (bucket-grads b)
                                   (move-if-needed scale home (bucket-device b))))
       total])))

(define/contract-out (clip-grad-value! params clip-value #:foreach? [foreach? #t]) ;; noqa
  (->* ((listof tensor?) (and/c real? (>=/c 0))) (#:foreach? boolean?) void?)
  (define v (exact->inexact clip-value))
  (define kernels (kernels-for foreach?))
  (with-gradients ([qs gs] params)
    (for ([b (in-list (buckets-of qs gs))])
      ((kernels-clamp! kernels) (bucket-grads b) (- v) v))))

(struct kernels (norms scale! clamp!))

(define foreach-kernels
  (kernels (lambda (grads ord) (g:foreach-norm-scalar grads ord #f))
           g:foreach-mul-tensor!
           (lambda (grads lo hi)
             (g:foreach-clamp-min-scalar! grads lo)
             (g:foreach-clamp-max-scalar! grads hi))))

(define loop-kernels
  (kernels (lambda (grads ord)
             (for/list ([g (in-list grads)]) (g:linalg-vector-norm g ord #f #f #f)))
           (lambda (grads scale)
             (for ([g (in-list grads)]) (g:mul-tensor! g scale)))
           (lambda (grads lo hi)
             (for ([g (in-list grads)]) (copy! g (clamp g #:min lo #:max hi))))))

(define (kernels-for foreach?)
  (if foreach? foreach-kernels loop-kernels))

(struct placement (device dtype) #:transparent)

(struct entry (placement grad))

(struct bucket (device grads))

(struct placed (generation placement))

(define placements (make-weak-hasheq))

(define (placement-of p)
  (define now (move-generation))
  (match (hash-ref placements p #f)
    [(placed (== now eqv?) known) known]
    [_
     (define fresh (placement (device p) (dtype p)))
     (hash-set! placements p (placed now fresh))
     fresh]))

(define (buckets-of params grads)
  (for/list ([entries (in-list (group-by entry-placement
                                         (map entry (map placement-of params) grads)))])
    (bucket (~> entries first entry-placement placement-device)
            (map entry-grad entries))))

(define (move-if-needed t from to)
  (if (equal? from to) t (to-device t to)))

(define (joined norms)
  (match norms
    [(list one) one]
    [_ (cat norms)]))

(define (total-norm buckets home ord kernels)
  (define (stacked b)
    (~> (bucket-grads b)
        ((kernels-norms kernels) ord)
        stack
        (move-if-needed (bucket-device b) home)))
  (~> (map stacked buckets)
      joined
      (g:linalg-vector-norm ord #f #f #f)))
