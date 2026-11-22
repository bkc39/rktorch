#lang racket/base

(provide not-implemented)

(define legs #hasheq((T1 . "T1 (#384)") (T2 . "T2 (#385)") (T3 . "T3 (#386)")))

(define (not-implemented who leg)
  (raise (exn:fail:unsupported
          (format "~a: not implemented until ~a" who (hash-ref legs leg))
          (current-continuation-marks))))
