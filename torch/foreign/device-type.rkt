#lang racket/base

(provide (struct-out device)
         cpu-device
         cuda-device)

(struct device (type index)
  #:transparent
  #:guard (lambda (type index name)
            (unless (memq type '(cpu cuda))
              (error name "unsupported device type: ~e" type))
            (unless (exact-nonnegative-integer? index)
              (error name "index must be an exact nonnegative integer: ~e"
                     index))
            (when (and (eq? type 'cpu) (not (zero? index)))
              (error name "cpu device index must be 0: ~e" index))
            (values type index))
  #:property prop:custom-write
  (lambda (d port _mode)
    (if (eq? (device-type d) 'cpu)
        (fprintf port "#<device cpu>")
        (fprintf port "#<device cuda:~a>" (device-index d)))))

(define (cpu-device) (device 'cpu 0))

(define (cuda-device [index 0]) (device 'cuda index))
