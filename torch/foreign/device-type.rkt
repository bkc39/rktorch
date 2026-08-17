#lang racket/base

;; First-class device values (PR B-0 of plans/gpu-memory-management.md): a
;; transparent struct, so equal?/equal-hash work field-wise — device values
;; become the #37 memory ledger's hash keys — with a torch.device-style
;; printed form. The legacy symbol/list forms ('cpu, 'cuda,
;; (list 'cuda n)) stay accepted everywhere a device argument is taken (the
;; PyTorch pattern: device objects beside legacy strings), but queries
;; (tensor-device / default-device) return these structs.

(provide (struct-out device)
         cpu-device
         cuda-device)

;; type: 'cpu | 'cuda; index: an exact nonnegative ordinal (0 for cpu —
;; there is only one). The guard makes malformed devices unrepresentable
;; rather than deferring to a downstream marshalling error.
(struct device (type index)
  #:transparent
  #:guard (lambda (type index name)
            (unless (memq type '(cpu cuda))
              (error name "unsupported device type: ~e" type))
            (unless (exact-nonnegative-integer? index)
              (error name "index must be an exact nonnegative integer: ~e"
                     index))
            (values type index))
  #:property prop:custom-write
  (lambda (d port _mode)
    (if (eq? (device-type d) 'cpu)
        (fprintf port "#<device cpu>")
        (fprintf port "#<device cuda:~a>" (device-index d)))))

(define (cpu-device) (device 'cpu 0))

(define (cuda-device [index 0]) (device 'cuda index))
