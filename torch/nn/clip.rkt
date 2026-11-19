#lang racket/base

(require (only-in racket/contract/base
                  ->* >=/c and/c flat-named-contract listof not/c)
         (only-in racket/match match)
         (only-in racket/math nan?)
         (only-in "../foreign.rkt"
                  * + clamp device div sqrt sum tensor? to-device with-no-grad
                  zeros)
         (only-in "../foreign/autograd-ops.rkt" with-gradients)
         (only-in "../generated.rkt" [mul-tensor! g:mul-tensor!])
         (only-in "../private/contract.rkt" define/contract-out))

(define norm-type/c (flat-named-contract 'norm-type/c (and/c real? (not/c nan?))))

(define/contract-out (clip-grad-norm! params max-norm ;; noqa
                                      #:norm-type [norm-type 2.0]
                                      #:error-if-nonfinite? [error-if-nonfinite? #f]
                                      #:foreach? [foreach? #t])
  (->* ((listof tensor?) (and/c real? (>=/c 0)))
       (#:norm-type norm-type/c #:error-if-nonfinite? boolean? #:foreach? boolean?)
       tensor?)
  (cond
    [(or (not (= norm-type 2)) error-if-nonfinite? (not foreach?)) (error 'unimplemented)]
    [else (l2-clip! params max-norm)]))

(define/contract-out (clip-grad-value! params clip-value #:foreach? [foreach? #t]) ;; noqa
  (->* ((listof tensor?) (and/c real? (>=/c 0))) (#:foreach? boolean?) void?)
  (error 'unimplemented))

;; The scale stays a tensor from the norm to the multiply, as in
;; torch.nn.utils.clip_grad_norm_, so clipping never waits on the device.
(define (l2-clip! params max-norm)
  (with-no-grad
    (with-gradients (grads params)
      (match grads
        ['() (zeros)]
        [(cons leading others)
         (define home (device leading))
         (define total
           (sqrt (for/fold ([acc (sum (* leading leading))])
                           ([g (in-list others)])
                   (+ acc (to-device (sum (* g g)) home)))))
         (define scale (clamp (div max-norm (+ total 1e-6)) #:max 1.0))
         (for ([g (in-list grads)])
           (g:mul-tensor! g (to-device scale (device g))))
         total]))))
