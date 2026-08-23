#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in ffi/unsafe
                  _double _enum _fun _int _int64 _ptr _void
                  register-finalizer)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in "../device-type.rkt" device device-index device-type)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-tensor-free/finalizer
         tr-tensor-free/checked
         collect-and-drain!
         swallow-and-count-failure
         finalizer-failures
         tensor-allocator
         tensor-allocator/rng
         oom-retry
         tr-cuda-empty-cache/raw
         tr-last-error-kind/raw
         native-memory-use
         _tr-device-type ;; noqa
         tr-tensor-device/raw
         define-unary/raw
         define-binary/raw
         define-scalar/raw)

(define-torch tr-tensor-free/unwrapped
  (_fun _Tensor -> _void)
  #:c-id tr_tensor_free)

;; The (deallocator) wrap cancels the pending GC finalizer.
(define tr-tensor-free/checked
  (let ([release ((deallocator) tr-tensor-free/unwrapped)])
    (lambda (t)
      (unaccount! t)
      (release t))))

(define finalizer-failure-count (box 0))

(define (finalizer-failures)
  (unbox finalizer-failure-count))

;; Total catch on purpose, not exn:fail?: any value escaping GC
;; finalization re-enters the error machinery and cascades (#38).
(define ((swallow-and-count-failure release) t)
  (with-handlers ([(lambda (_) #t)
                   (lambda (_e)
                     (call-with-ledger
                      (lambda ()
                        (set-box! finalizer-failure-count
                                  (add1 (unbox finalizer-failure-count))))))])
    (release t)))

(struct allocation (phantom nbytes device))

(define allocations (make-weak-hasheq))

;; Atomic mode, not a semaphore: finalizers run in atomic mode, where
;; blocking is an internal error.
(define (call-with-ledger thunk)
  (call-as-atomic thunk))

(define-torch tr-tensor-nbytes/raw
  (_fun _Tensor (out : (_ptr o _int64)) -> (rc : _int) -> (values rc out))
  #:c-id tr_tensor_nbytes)

(define _tr-device-type
  (_enum '(cpu = 0 cuda = 1 mps = 2)))

(define-torch tr-tensor-device/raw
  (_fun _Tensor
        (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_tensor_device)

(define (account! t)
  (with-handlers ([exn:fail? void])
    (define-values (nb-rc nbytes) (tr-tensor-nbytes/raw t))
    (define-values (dev-rc type index) (tr-tensor-device/raw t))
    (when (and (zero? nb-rc) (zero? dev-rc))
      (define entry
        (allocation (make-phantom-bytes nbytes)
                    nbytes
                    (device type (if (eq? type 'cpu) 0 index))))
      (call-with-ledger (lambda () (hash-set! allocations t entry))))))

(define (unaccount! t)
  (with-handlers ([exn:fail? void])
    (call-with-ledger
     (lambda ()
       (define a (hash-ref allocations t #f))
       (when a
         (set-phantom-bytes! (allocation-phantom a) 0)
         (hash-remove! allocations t))))))

(define (native-memory-use)
  (define entries (call-with-ledger (lambda () (hash-values allocations))))
  (define totals (make-hash))
  (for ([a (in-list entries)])
    (hash-update! totals (allocation-device a)
                  (lambda (n) (+ n (allocation-nbytes a)))
                  0))
  (sort (hash->list totals)
        (lambda (x y)
          (define dx (car x))
          (define dy (car y))
          (cond
            [(eq? (device-type dx) (device-type dy))
             (< (device-index dx) (device-index dy))]
            [else (eq? (device-type dx) 'cpu)]))))

;; Unwrapped release on purpose: the (deallocator) wrap would cancel the
;; very registration this finalizer runs from.
(define tr-tensor-free/finalizer
  (swallow-and-count-failure
   (lambda (t)
     (unaccount! t)
     (tr-tensor-free/unwrapped t))))

(define-torch tr-last-error-kind/raw
  (_fun -> _int)
  #:c-id tr_last_error_kind)

(define (last-error-oom?)
  (= 1 (tr-last-error-kind/raw)))

(define-torch tr-cuda-empty-cache/raw
  (_fun -> _int)
  #:c-id tr_cuda_empty_cache)

(define (collect-and-drain!)
  (define canary-finalized (make-semaphore 0))
  (register-finalizer (box 0) (lambda (_) (semaphore-post canary-finalized)))
  (collect-garbage)
  (define observed (sync/timeout 0.5 canary-finalized))
  (void (tr-cuda-empty-cache/raw))
  (and observed #t))

(define ((oom-retry #:oom? [oom? last-error-oom?]
                    #:collect! [collect! collect-and-drain!])
         raw-fn)
  (lambda args
    (define t (apply raw-fn args))
    (cond
      [t t]
      [(oom?)
       (collect!)
       (apply raw-fn args)]
      [else #f])))

(define ((accounted wrapped) . args)
  (define t (apply wrapped args))
  (when t (account! t))
  t)

;; The retry composes OUTSIDE the allocator wrap: ffi/unsafe/alloc runs
;; the wrapped call in atomic mode, where the drain's blocking wait is an
;; internal error.
(define (tensor-allocator raw-fn)
  (accounted ((oom-retry) ((allocator tr-tensor-free/finalizer) raw-fn))))

;; No retry: these bindings consume the global RNG stream, and a blind
;; retry would draw twice and break seeded parity.
(define (tensor-allocator/rng raw-fn)
  (accounted ((allocator tr-tensor-free/finalizer) raw-fn)))

(define-syntax (define-unary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (t : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

(define-syntax (define-binary/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _Tensor) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))

(define-syntax (define-scalar/raw stx)
  (syntax-parse stx
    [(_ name:id c-id:id)
     #'(define-torch name
         (_fun (a : _Tensor) (b : _double) -> _Tensor/null)
         #:c-id c-id
         #:wrap tensor-allocator)]))
