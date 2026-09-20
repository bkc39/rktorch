#lang racket/base

(require (only-in racket/contract/base -> and/c listof >=/c)
         (only-in racket/list filter-map)
         (only-in "../foreign.rkt"
                  * + clamp div maybe-grad sqrt sum tensor-device tensor?
                  to-device with-no-grad zeros)
         (only-in "../generated.rkt" [mul-tensor! g:mul-tensor!])
         (only-in "../private/contract.rkt" define/contract-out))

;; The scale stays a tensor from the norm to the multiply, as in
;; torch.nn.utils.clip_grad_norm_, so clipping never waits on the device.
(define/contract-out (clip-grad-norm! params max-norm) ;; noqa
  (-> (listof tensor?) (and/c real? (>=/c 0)) tensor?)
  (with-no-grad
    (define grads (filter-map maybe-grad params))
    (cond
      [(null? grads) (zeros)]
      [else
       (define home (tensor-device (car grads)))
       (define total
         (sqrt (for/fold ([acc (sum (* (car grads) (car grads)))])
                         ([g (in-list (cdr grads))])
                 (+ acc (to-device (sum (* g g)) home)))))
       (define scale (clamp (div max-norm (+ total 1e-6)) #:max 1.0))
       (for ([g (in-list grads)])
         (g:mul-tensor! g (to-device scale (tensor-device g))))
       total])))
