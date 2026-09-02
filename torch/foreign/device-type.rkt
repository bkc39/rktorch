#lang racket/base

(require (only-in racket/contract/base -> ->* any/c contract-out list/c or/c)
         (only-in "../private/contract.rkt" define/checked-out))

(provide (struct-out device)
         device/c)

(struct device (type index)
  #:transparent
  #:guard (lambda (type index name)
            (unless (memq type '(cpu cuda mps))
              (error name "unsupported device type: ~e" type))
            (unless (exact-nonnegative-integer? index)
              (error name "index must be an exact nonnegative integer: ~e"
                     index))
            (when (and (memq type '(cpu mps)) (not (zero? index)))
              (error name "~a device index must be 0: ~e" type index))
            (values type index))
  #:property prop:custom-write
  (lambda (d port _mode)
    (if (eq? (device-type d) 'cuda)
        (fprintf port "#<device cuda:~a>" (device-index d))
        (fprintf port "#<device ~a>" (device-type d)))))

(module+ checked
  (provide (contract-out
            [device? (-> any/c boolean?)]
            [device-type (-> device? (or/c 'cpu 'cuda 'mps))]
            [device-index (-> device? exact-nonnegative-integer?)])))

(define device/c
  (or/c device? 'cpu 'cuda 'mps (list/c 'cuda exact-nonnegative-integer?)))

(define/checked-out (cpu-device) (-> device?) (device 'cpu 0))

(define/checked-out (cuda-device [index 0])
  (->* [] [exact-nonnegative-integer?] device?)
  (device 'cuda index))

(define/checked-out (mps-device) (-> device?) (device 'mps 0))
