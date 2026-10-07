#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in ffi/unsafe _double _enum _fun _int _int64 _ptr _void)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in racket/list remove-duplicates)
         (only-in "../device-type.rkt" device device-index device-type)
         (only-in "collector.rkt"
                  collect-and-wait!
                  finalizer-runs
                  note-finalizer-run!)
         (only-in "fault.rkt" native-fault? note-native-fault!)
         (only-in "pressure.rkt"
                  call-with-ledger
                  collect-under-pressure!
                  live-bytes-by-device
                  lower-shadows!
                  note-accounted!
                  note-adopted!
                  note-unaccounted!
                  note-unadopted!
                  pressure-diagnostics
                  shadow-generation
                  unaccounted-bytes-by-device)
         (only-in "race-points.rkt" race-point)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-tensor-free/finalizer
         tr-tensor-free/checked
         collect-and-drain!
         swallow-and-count-failure
         finalizer-failures
         finalizer-diagnostics
         tensor-allocator
         tensor-allocator/gradient
         tensor-allocator/no-retry
         tensor-allocator/outputs
         tensor-allocator/outputs/no-retry
         oom-retry
         oom-retry/status
         reaccount!
         record-allocation!
         unaccount!
         tr-cuda-empty-cache/raw
         tr-mps-empty-cache/raw
         tr-last-error-kind/raw
         native-memory-use
         native-memory-use/fold
         native-memory-unaccounted
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
      (race-point free-unaccounted t)
      (release t))))

(define finalizer-failure-count (box 0))
(define captured-failures (box '()))
(define capture-limit 8)

(define (finalizer-failures)
  (unbox finalizer-failure-count))

(define (finalizer-diagnostics)
  (call-with-ledger
   (lambda ()
     (append (list (cons 'runs (finalizer-runs))
                   (cons 'failures (unbox finalizer-failure-count))
                   (cons 'messages (reverse (unbox captured-failures)))
                   (cons 'ledger-entries (hash-count allocations)))
             (pressure-diagnostics)))))

;; No printer: a prop:custom-write that raises or blocks would be fatal here.
(define (describe-raised e)
  (cond
    [(exn? e) (exn-message e)]
    [(symbol? e) (symbol->string e)]
    [(string? e) e]
    [else "non-exn value raised from a native release"]))

(define (take-at-most n xs)
  (cond
    [(or (zero? n) (null? xs)) '()]
    [else (cons (car xs) (take-at-most (sub1 n) (cdr xs)))]))

;; Guarded: this is the handler below, so nothing else protects it.
(define (record-failure! e)
  (with-handlers ([(lambda (_) #t) void])
    (when (native-fault? e)
      (note-native-fault! "running a finalizer"))
    (record-failure!/unguarded e)))

(define (record-failure!/unguarded e)
  (call-with-ledger
   (lambda ()
     (set-box! finalizer-failure-count (add1 (unbox finalizer-failure-count)))
     (set-box! captured-failures
               (take-at-most capture-limit
                             (cons (describe-raised e)
                                   (unbox captured-failures)))))))

;; Total catch on purpose, not exn:fail?: any value escaping GC
;; finalization re-enters the error machinery and cascades (#38).
(define ((swallow-and-count-failure release) t)
  ;; Nothing outside the guard: alloc.rkt's finalizer holds raw atomic mode
  ;; with no dynamic-wind, so an escape kills the process rather than raising.
  (with-handlers ([(lambda (_) #t) record-failure!])
    (call-with-ledger note-finalizer-run!)
    (release t)))

(struct allocation (phantom nbytes device [adopted #:mutable]))

(define allocations (make-weak-hasheq))

(define-torch tr-tensor-nbytes/raw
  (_fun _Tensor (out : (_ptr o _int64)) -> (rc : _int) -> (values rc out))
  #:c-id tr_tensor_nbytes)

(define _tr-device-type
  (_enum '(cpu = 0 cuda = 1 mps = 2 keep = -1) _int))

(define-torch tr-tensor-device/raw
  (_fun _Tensor
        (type : (_ptr o _tr-device-type))
        (index : (_ptr o _int64))
        -> (rc : _int)
        -> (values rc type index))
  #:c-id tr_tensor_device)

(define (account! t)
  (with-handlers ([exn:fail? (lambda (e)
                               (when (native-fault? e)
                                 (note-native-fault! "accounting a tensor"))
                               #f)])
    (define-values (nb-rc nbytes) (tr-tensor-nbytes/raw t))
    (define-values (dev-rc type index) (tr-tensor-device/raw t))
    (and (zero? nb-rc)
         (zero? dev-rc)
         (let ([dev (device type (if (eq? type 'cpu) 0 index))])
           (record-allocation! t nbytes dev)
           dev))))

(define (record-allocation! t nbytes dev)
  (define entry (allocation (make-phantom-bytes nbytes) nbytes dev 0))
  (call-with-ledger
   (lambda ()
     (hash-set! allocations t entry)
     (race-point ledger-entry-added t)
     (note-accounted! dev nbytes))))

(define (unaccount! t)
  (with-handlers ([exn:fail? void])
    (call-with-ledger
     (lambda ()
       (define a (hash-ref allocations t #f))
       (when a
         (race-point unaccount-entry-read t)
         (set-phantom-bytes! (allocation-phantom a) 0)
         (hash-remove! allocations t)
         (note-unaccounted! (allocation-device a) (allocation-nbytes a))
         ;; the parameter still holds the gradient this handle took over
         (when (positive? (allocation-adopted a))
           (note-unadopted! (allocation-device a) (allocation-adopted a))))))))

;; An in-place move (tr_tensor_to_) changes the device and byte count under
;; the same handle, so its ledger entry is replaced rather than added to. When
;; the new size cannot be read the old charge goes back: a stale entry still
;; presses on the collector, a missing one would not.
(define (reaccount! t)
  (define old (call-with-ledger (lambda () (hash-ref allocations t #f))))
  (unaccount! t)
  (define dev (account! t))
  (cond
    [dev (collect-under-pressure! dev)]
    [old (record-allocation! t (allocation-nbytes old) (allocation-device old))]
    [else (void)]))

(define (sort-by-device totals)
  (sort totals
        (lambda (x y)
          (define dx (car x))
          (define dy (car y))
          (cond
            [(eq? (device-type dx) (device-type dy))
             (< (device-index dx) (device-index dy))]
            [else (eq? (device-type dx) 'cpu)]))))

(define (positive-by-device totals)
  (sort-by-device (filter (lambda (entry) (positive? (cdr entry))) totals)))

(define (native-memory-use)
  (positive-by-device (live-bytes-by-device)))

(define (native-memory-unaccounted)
  (positive-by-device (unaccounted-bytes-by-device)))

;; The entry-by-entry fold; the counters above must agree with it.
(define (native-memory-use/fold)
  (define entries (call-with-ledger (lambda () (hash-values allocations))))
  (define totals (make-hash))
  (for ([a (in-list entries)])
    (hash-update! totals (allocation-device a)
                  (lambda (n) (+ n (allocation-nbytes a)))
                  0))
  (sort-by-device (hash->list totals)))

;; Unwrapped release on purpose: the (deallocator) wrap would cancel the
;; very registration this finalizer runs from.
(define tr-tensor-free/finalizer
  (swallow-and-count-failure
   (lambda (t)
     (race-point finalizer-releasing t)
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
    [(and (failed? result) (oom?))
     (collect!)
     (apply raw-fn args)]
    [else result]))

;; handle-returning raw calls: #f is the failure
(define (oom-retry #:oom? [oom? last-error-oom?]
                   #:collect! [collect! collect-and-drain!])
  (retry-on-oom not oom? collect!))

;; status-returning raw calls: 1 is the failure
(define (oom-retry/status #:oom? [oom? last-error-oom?]
                          #:collect! [collect! collect-and-drain!])
  (retry-on-oom (lambda (rc) (= rc 1)) oom? collect!))

;; `adopt` runs after the accounting and before the pressure check, so the
;; check sees the charge where it will stay.
(define (((accounted adopt) wrapped) . args)
  (define t (apply wrapped args))
  (race-point op-returned t)
  (when t
    (define dev (account! t))
    (when dev
      (adopt args t)
      (collect-under-pressure! dev)))
  t)

(define (adopt-nothing _args _t)
  (void))

;; A gradient is storage the backward pass wrote before any handle pointed
;; at it, so the shadow's last refresh may charge it. The first handle taken
;; on a parameter's gradient since that refresh moves those bytes to the
;; ledger; a later one, which the ledger charges again as it does any second
;; handle, moves nothing.
(define adopted-at (make-weak-hasheq))

(define (adopt-gradient! args t)
  (define param (car args))
  (call-with-ledger
   (lambda ()
     (define now (shadow-generation))
     (define entry (hash-ref allocations t #f))
     (when (and entry (not (eqv? (hash-ref adopted-at param #f) now)))
       (hash-set! adopted-at param now)
       (set-allocation-adopted!
        entry
        (note-adopted! (allocation-device entry) (allocation-nbytes entry)))))))

;; The retry composes OUTSIDE the allocator wrap: ffi/unsafe/alloc runs
;; the wrapped call in atomic mode, where the drain's blocking wait is an
;; internal error.
(define (tensor-allocator raw-fn)
  ((accounted adopt-nothing)
   ((oom-retry) ((allocator tr-tensor-free/finalizer) raw-fn))))

;; for tr_tensor_grad, whose first argument is the parameter
(define (tensor-allocator/gradient raw-fn)
  ((accounted adopt-gradient!)
   ((oom-retry) ((allocator tr-tensor-free/finalizer) raw-fn))))

;; No retry: re-running these after an OOM would repeat something the
;; first call already did — a draw from the global RNG stream, or an
;; in-place update of a tensor the caller handed in.
(define (tensor-allocator/no-retry raw-fn)
  ((accounted adopt-nothing) ((allocator tr-tensor-free/finalizer) raw-fn)))

(define adopt-handle ((allocator tr-tensor-free/finalizer) values))

;; One atomic section spans the call and every registration, as (allocator)
;; does for a single result, so no break lands between a handle and its
;; finalizer.
(define ((adopting-outputs raw-fn) . args)
  (call-as-atomic
   (lambda ()
     (define handles (apply raw-fn args))
     (and handles (map adopt-handle handles)))))

(define ((accounted-outputs wrapped) . args)
  (define handles (apply wrapped args))
  (when handles
    ;; every output is on the ledger before any collection measures it
    (for ([dev (in-list (remove-duplicates (filter values
                                                   (map account! handles))))])
      (collect-under-pressure! dev)))
  handles)

;; For raw calls answering a list of handles, or #f on failure.
(define (tensor-allocator/outputs raw-fn)
  (accounted-outputs ((oom-retry) (adopting-outputs raw-fn))))

(define (tensor-allocator/outputs/no-retry raw-fn)
  (accounted-outputs (adopting-outputs raw-fn)))

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

;; No I/O from inside a finalizer -- that runs in atomic mode, where writing to
;; a port can block and trip an internal error.  Accumulate, dump here.
(define mem-trace-handle
  (and (getenv "RKTORCH_MEM_TRACE")
       (plumber-add-flush!
        (current-plumber)
        (lambda (_h)
          (with-handlers ([(lambda (_) #t) void])
            (eprintf "[rktorch mem] ~s\n" (finalizer-diagnostics)))))))
