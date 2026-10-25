#lang racket/base

(require (only-in racket/contract/base -> and/c listof >=/c)
         (only-in racket/match match)
         (only-in "../foreign.rkt"
                  * + clamp device div sqrt sum tensor? to-device with-no-grad
                  zeros)
         (only-in "../foreign/autograd-ops.rkt" with-gradients)
         (only-in "../generated.rkt" [mul-tensor! g:mul-tensor!])
         (only-in "../private/contract.rkt" define/contract-out))

;; The scale stays a tensor from the norm to the multiply, as in
;; torch.nn.utils.clip_grad_norm_, so clipping never waits on the device.
(define/contract-out (clip-grad-norm! params max-norm) ;; noqa
  (-> (listof tensor?) (and/c real? (>=/c 0)) tensor?)
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
