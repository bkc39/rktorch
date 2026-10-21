#lang racket/base

(require (only-in racket/contract/base ->)
         (only-in racket/promise delay/sync force)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "creation-ops.rkt" zeros)
         (only-in "device-type.rkt" cpu-device cuda-device device? mps-device)
         (only-in "ops.rkt" cuda-available? mps-available? to-device))

(provide mps-usable
         pick-accelerator)

(define-logger rktorch-device)

(define (probe-mps)
  (to-device (zeros 1 #:device (mps-device)) (cpu-device)))

(define (mps-usable [probe probe-mps])
  (delay/sync
   (with-handlers ([exn:fail?
                    (λ (e)
                      (log-rktorch-device-warning
                       "MPS is available but cannot allocate, so the CPU stands in: ~a"
                       (exn-message e))
                      #f)])
     (probe)
     #t)))

;; CUDA before MPS only for determinism; the two never coexist (MPS is
;; darwin-only, and nixpkgs has no darwin CUDA libtorch).
(define (pick-accelerator cuda? mps? usable)
  (cond
    [cuda? (cuda-device)]
    [(and mps? (force usable)) (mps-device)]
    [else (cpu-device)]))

(define usable-here (mps-usable))

(define/contract-out (mps-if-available) (-> device?) ;; noqa
  (pick-accelerator #f (mps-available?) usable-here))

(define/contract-out (accelerator-if-available) (-> device?) ;; noqa
  (pick-accelerator (cuda-available?) (mps-available?) usable-here))
