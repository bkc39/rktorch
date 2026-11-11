#lang racket/base

(require (only-in racket/contract/base
                  ->* >=/c and/c flat-named-contract listof not/c)
         (only-in racket/list group-by)
         (only-in racket/match == match match-define)
         (only-in racket/math nan?)
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
  (with-gradients ([qs gs] params)
    (match (buckets-of qs gs)
      ['() (zeros)]
      [(and buckets (cons (bucket home _) _))
       (define total (total-norm buckets home ord foreach?))
       (when (and error-if-nonfinite? (not (rational? (item total))))
         (error 'clip-grad-norm!
                (string-append "the total norm of order ~a of the gradients is non-finite, "
                               "so it cannot be clipped")
                ord))
       (define scale (clamp (/ max-norm (+ total 1e-6)) #:max 1.0))
       (for ([b (in-list buckets)])
         (match-define (bucket dev grads) b)
         (define here (moved scale dev home))
         (if foreach?
             (g:foreach-mul-tensor! grads here)
             (for ([g (in-list grads)]) (g:mul-tensor! g here))))
       total])))

(define/contract-out (clip-grad-value! params clip-value #:foreach? [foreach? #t]) ;; noqa
  (->* ((listof tensor?) (and/c real? (>=/c 0))) (#:foreach? boolean?) void?)
  (define v (exact->inexact clip-value))
  (with-gradients ([qs gs] params)
    (for ([b (in-list (buckets-of qs gs))])
      (match-define (bucket _ grads) b)
      (cond
        [foreach?
         (g:foreach-clamp-min-scalar! grads (- v))
         (g:foreach-clamp-max-scalar! grads v)]
        [else
         (for ([g (in-list grads)])
           (copy! g (clamp g #:min (- v) #:max v)))]))))

(struct bucket (device grads))

(struct placed (generation key))

(define placements (make-weak-hasheq))

(define (placement p)
  (define now (move-generation))
  (match (hash-ref placements p #f)
    [(placed (== now eqv?) key) key]
    [_
     (define key (cons (device p) (dtype p)))
     (hash-set! placements p (placed now key))
     key]))

(define (buckets-of params grads)
  (define rows (map (lambda (p g) (cons (placement p) g)) params grads))
  (for/list ([group (in-list (group-by car rows))])
    (match-define (list (cons (cons devs _) grads) ...) group)
    (bucket (car devs) grads)))

(define (moved t target from)
  (if (equal? target from) t (to-device t target)))

(define (bucket-norms b ord foreach?)
  (match-define (bucket _ grads) b)
  (if foreach?
      (g:foreach-norm-scalar grads ord #f)
      (for/list ([g (in-list grads)]) (g:linalg-vector-norm g ord #f #f #f))))

(define (total-norm buckets home ord foreach?)
  (define (stacked b)
    (stack (bucket-norms b ord foreach?)))
  (define norms
    (match buckets
      [(list one) (stacked one)]
      [_ (cat (for/list ([b (in-list buckets)])
                (moved (stacked b) home (bucket-device b))))]))
  (g:linalg-vector-norm norms ord #f #f #f))
