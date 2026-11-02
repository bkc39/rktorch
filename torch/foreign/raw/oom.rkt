#lang racket/base

(require (only-in ffi/unsafe _fun _int)
         (only-in "collector.rkt" collect-and-wait!)
         (only-in "pressure.rkt" lower-shadows!)
         (only-in "race-points.rkt" race-point)
         (only-in "syntax.rkt" define-torch))

(provide collect-and-drain!
         oom-retry
         oom-retry/status
         tr-cuda-empty-cache/raw
         tr-last-error-kind/raw
         tr-mps-empty-cache/raw)

(define-torch tr-last-error-kind/raw
  (_fun -> _int)
  #:c-id tr_last_error_kind)

(define (last-error-oom?)
  (= 1 (tr-last-error-kind/raw)))

(define-torch tr-cuda-empty-cache/raw
  (_fun -> _int)
  #:c-id tr_cuda_empty_cache)

(define-torch tr-mps-empty-cache/raw
  (_fun -> _int)
  #:c-id tr_mps_empty_cache)

(define (collect-and-drain!)
  (define-values (observed _drained?) (collect-and-wait!))
  (void (tr-cuda-empty-cache/raw))
  (void (tr-mps-empty-cache/raw))
  (lower-shadows!)
  observed)

;; one retry after a collect when a failed call was an OOM; the two
;; wrappers below differ only in how a raw result reports failure
(define (((retry-on-oom failed? oom? collect!) raw-fn) . args)
  (define result (apply raw-fn args))
  (cond
    [(and (failed? result) (oom-kind? oom? raw-fn))
     (collect!)
     (apply raw-fn args)]
    [else result]))

(define (oom-kind? oom? raw-fn)
  (race-point oom-kind-reading raw-fn)
  (oom?))

;; handle-returning raw calls: #f is the failure
(define (oom-retry #:oom? [oom? last-error-oom?]
                   #:collect! [collect! collect-and-drain!])
  (retry-on-oom not oom? collect!))

;; status-returning raw calls: 1 is the failure
(define (oom-retry/status #:oom? [oom? last-error-oom?]
                          #:collect! [collect! collect-and-drain!])
  (retry-on-oom (lambda (rc) (= rc 1)) oom? collect!))
