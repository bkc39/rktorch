#lang racket/base

(require (only-in ffi/vector
                  f32vector->list
                  f64vector->list
                  f64vector?
                  list->s64vector
                  make-f32vector
                  make-f64vector
                  make-s64vector
                  s64vector->list
                  s64vector-ref
                  s64vector?)
         (only-in "device-type.rkt"
                  [device make-device]
                  cpu-device cuda-device mps-device
                  device-index device-type device?)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/device.rkt"
                  tr-cuda-device-count/raw
                  tr-cuda-empty-cache/raw
                  tr-cuda-is-available/raw
                  tr-cuda-memory-stats/raw
                  tr-get-default-device/raw
                  tr-mps-is-available/raw
                  tr-set-default-device/raw
                  tr-tensor-device/raw
                  tr-tensor-to-device/raw)
         (only-in "raw/global.rkt" tr-manual-seed/raw tr-version/raw)
         (only-in "raw/memory.rkt"
                  collect-and-drain!
                  finalizer-failures
                  native-memory-use)
         (only-in "raw/random.rkt" tr-rand/raw tr-randn/raw tr-tensor-uniform!/raw)
         (only-in "raw/tensor.rkt"
                  dtype-code->symbol
                  tr-tensor-copy-data-f64/raw
                  tr-tensor-copy-data-i64/raw
                  tr-tensor-copy-data/raw
                  tr-tensor-dtype/raw
                  tr-tensor-item/raw
                  tr-tensor-numel/raw
                  tr-tensor-to-dtype/raw)
         (only-in "structs.rkt"
                  handle->repr
                  handle->string
                  tensor-handle
                  tensor-impl-shape
                  tensor?
                  wrap-tensor))

(provide torch-version
         cuda-empty-cache!
         cuda-memory-stats
         reclaim-native-memory!
         device
         device->type+index
         dtype
         numel
         shape
         finalizer-failures
         native-memory-use
         manual-seed!
         randn
         rand
         uniform!
         item
         tensor-dtype
         to-dtype
         cuda-available?
         cuda-if-available
         cuda-device-count
         mps-available?
         mps-if-available
         set-default-device!
         default-device
         call-with-default-device
         with-default-device
         to-device
         tensor-device
         tensor-numel
         tensor-shape
         tensor->vector
         tensor->list
         tensor->string
         tensor->repr)

(define (torch-version)
  (tr-version/raw))

(define (manual-seed! seed)
  (check-ok (tr-manual-seed/raw seed) 'manual-seed!)
  (void))

(define (randn . dims)
  (wrap-tensor
   (check-handle 'randn (tr-randn/raw (list->s64vector dims) (length dims)))))

(define (rand . dims)
  (wrap-tensor
   (check-handle 'rand (tr-rand/raw (list->s64vector dims) (length dims)))))

(define (uniform! t low high)
  (check-ok (tr-tensor-uniform!/raw t
                                    (exact->inexact low)
                                    (exact->inexact high))
            'uniform!)
  (void))

(define (item t)
  (cond
    [(and (int64-tensor? t) (= 1 (tensor-numel t)))
     (define out (make-s64vector 1))
     (define-values (rc _n) (tr-tensor-copy-data-i64/raw t 1 out))
     (check-ok rc 'item)
     (s64vector-ref out 0)]
    [else
     (define-values (rc v) (tr-tensor-item/raw t))
     (check-ok rc 'item)
     v]))

(define (to-dtype t dtype)
  (wrap-tensor (check-handle 'to-dtype (tr-tensor-to-dtype/raw t dtype))))

(define (tensor-dtype t)
  (define-values (rc code) (tr-tensor-dtype/raw t))
  (check-ok rc 'tensor-dtype)
  (or (dtype-code->symbol code)
      (error 'tensor-dtype "unsupported dtype code: ~a" code)))

(define (int64-tensor? t)
  (define-values (rc code) (tr-tensor-dtype/raw t))
  (and (zero? rc) (eq? (dtype-code->symbol code) 'int64)))

(define (device->type+index dev)
  (cond
    [(device? dev) (values (device-type dev) (device-index dev))]
    [(eq? dev 'cpu) (values 'cpu 0)]
    [(eq? dev 'cuda) (values 'cuda 0)]
    [(eq? dev 'mps) (values 'mps 0)]
    [(pair? dev) (values 'cuda (cadr dev))]
    [else (error 'device "unsupported device: ~e" dev)]))

(define (type+index->device type index)
  (case type
    [(cpu) (cpu-device)]
    [(mps) (mps-device)]
    [else (cuda-device index)]))

(define (cuda-available?)
  (= 1 (tr-cuda-is-available/raw)))

(define (cuda-if-available)
  (if (cuda-available?) (cuda-device) (cpu-device)))

(define (mps-available?)
  (= 1 (tr-mps-is-available/raw)))

(define (mps-if-available)
  (if (mps-available?) (mps-device) (cpu-device)))

(define (cuda-device-count)
  (tr-cuda-device-count/raw))

(define (cuda-memory-stats [dev (cuda-device)])
  (define-values (type index) (device->type+index dev))
  (unless (eq? type 'cuda)
    (error 'cuda-memory-stats "expected a CUDA device, given: ~e" dev))
  (define-values (rc allocated reserved peak)
    (tr-cuda-memory-stats/raw index))
  (check-ok rc 'cuda-memory-stats)
  (list (cons 'allocated allocated)
        (cons 'reserved reserved)
        (cons 'peak-allocated peak)))

(define (cuda-empty-cache!)
  (check-ok (tr-cuda-empty-cache/raw) 'cuda-empty-cache!)
  (void))

(define (reclaim-native-memory!)
  (let loop ([prev (ledger-total)] [rounds 4])
    (define drained? (collect-and-drain!))
    (define now (ledger-total))
    ;; a busy executor can make zero ledger progress in one round without
    ;; being done, so no-progress only terminates after a drained round
    (when (and (> rounds 1)
               (or (< now prev) (not drained?)))
      (loop now (sub1 rounds))))
  (cuda-empty-cache!))

(define (ledger-total)
  (for/sum ([entry (in-list (native-memory-use))])
    (cdr entry)))

(define (set-default-device! dev)
  (define-values (type index) (device->type+index dev))
  (check-ok (tr-set-default-device/raw type index) 'set-default-device!)
  (void))

(define (default-device)
  (define-values (rc type index) (tr-get-default-device/raw))
  (check-ok rc 'default-device)
  (type+index->device type index))

(define (call-with-default-device dev thunk)
  (define saved (default-device))
  (dynamic-wind (lambda () (set-default-device! dev))
                thunk
                (lambda () (set-default-device! saved))))

(define-syntax-rule (with-default-device dev body ...)
  (call-with-default-device dev (lambda () body ...)))

(define (to-device t dev)
  (define-values (type index) (device->type+index dev))
  (wrap-tensor
   (check-handle 'to-device (tr-tensor-to-device/raw t type index))))

(define (tensor-device t)
  (define-values (rc type index) (tr-tensor-device/raw t))
  (check-ok rc 'tensor-device)
  (type+index->device type index))

(define (tensor-numel t)
  (define-values (rc n) (tr-tensor-numel/raw t))
  (check-ok rc 'tensor-numel)
  n)

(define (tensor-shape t)
  (tensor-impl-shape t))

(define shape tensor-shape)
(define dtype tensor-dtype)
(define numel tensor-numel)

(define (device x [index #f])
  (cond
    [(tensor? x)
     (when index
       (error 'device "an ordinal makes no sense when querying a tensor: ~e"
              index))
     (tensor-device x)]
    [else (make-device x (or index 0))]))

(define (tensor->vector t)
  (define n (tensor-numel t))
  (define-values (dtype-rc code) (tr-tensor-dtype/raw t))
  (define dt (and (zero? dtype-rc) (dtype-code->symbol code)))
  (cond
    [(eq? dt 'float64)
     (define out (make-f64vector n))
     (define-values (rc _n) (tr-tensor-copy-data-f64/raw t n out))
     (check-ok rc 'tensor->vector)
     out]
    [(eq? dt 'int64)
     (define out (make-s64vector n))
     (define-values (rc _n) (tr-tensor-copy-data-i64/raw t n out))
     (check-ok rc 'tensor->vector)
     out]
    [else
     (define out (make-f32vector n))
     (define-values (rc _n) (tr-tensor-copy-data/raw t n out))
     (check-ok rc 'tensor->vector)
     out]))

(define (tensor->list t)
  (define v (tensor->vector t))
  (cond
    [(s64vector? v) (s64vector->list v)]
    [(f64vector? v) (f64vector->list v)]
    [else (f32vector->list v)]))

(define (tensor->string t)
  (handle->string (tensor-handle t)))

(define (tensor->repr t)
  (handle->repr (tensor-handle t) (tensor-shape t)))
