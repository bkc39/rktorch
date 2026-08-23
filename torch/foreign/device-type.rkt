#lang racket/base

(provide (struct-out device)
         cpu-device
         cuda-device
         mps-device)

(struct device (type index)
  #:transparent
  #:guard (lambda (type index name)
            (unless (memq type '(cpu cuda mps))
              (error name "unsupported device type: ~e" type))
            (unless (exact-nonnegative-integer? index)
              (error name "index must be an exact nonnegative integer: ~e"
                     index))
            ;; cpu and mps are single devices; only cuda has ordinals
            (when (and (memq type '(cpu mps)) (not (zero? index)))
              (error name "~a device index must be 0: ~e" type index))
            (values type index))
  #:property prop:custom-write
  (lambda (d port _mode)
    (if (eq? (device-type d) 'cuda)
        (fprintf port "#<device cuda:~a>" (device-index d))
        (fprintf port "#<device ~a>" (device-type d)))))

(define (cpu-device) (device 'cpu 0))

(define (cuda-device [index 0]) (device 'cuda index))

(define (mps-device) (device 'mps 0))
