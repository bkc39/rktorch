#lang racket/base

;; First-class device values: a transparent struct, so equal?/equal-hash
;; work field-wise (devices are the memory ledger's hash keys), with a
;; torch.device-style printed form. The legacy 'cpu / 'cuda /
;; (list 'cuda n) forms stay accepted as arguments; queries return structs.

(provide (struct-out device)
         cpu-device
         cuda-device)

;; The guard makes malformed devices unrepresentable rather than deferring
;; to a downstream marshalling error.
(struct device (type index)
  #:transparent
  #:guard (lambda (type index name)
            (unless (memq type '(cpu cuda))
              (error name "unsupported device type: ~e" type))
            (unless (exact-nonnegative-integer? index)
              (error name "index must be an exact nonnegative integer: ~e"
                     index))
            ;; a nonzero cpu index would be rejected by set-default-device!
            ;; but silently dropped by to-device (the C++ sides differ)
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
