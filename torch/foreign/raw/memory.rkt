#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in ffi/unsafe
                  _double _enum _fun _int _int64 _ptr _void register-finalizer
                  set-cpointer-tag!)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match-define match-lambda)
         (only-in "../device-type.rkt" device device-index device-type)
         (only-in "accounts.rkt" with-ledger)
         (only-in "cas.rkt" cas-add! cas-claim! cas-claim-when! cas-update!)
         (only-in "collector.rkt"
                  finalizer-runs
                  note-finalizer-run!)
         (only-in "fault.rkt" native-fault? note-native-fault!)
         (only-in "oom.rkt"
                  collect-and-drain!
                  oom-retry
                  oom-retry/status
                  tr-cuda-empty-cache/raw
                  tr-last-error-kind/raw
                  tr-mps-empty-cache/raw)
         (only-in "pressure.rkt"
                  account-of
                  collect-under-pressure!
                  live-bytes-by-device
                  note-accounted!
                  note-adopted!
                  note-unaccounted!
                  note-unadopted!
                  pressure-diagnostics
                  shadow-generation
                  unaccounted-bytes-by-device)
         (only-in "race-points.rkt" race-point)
         (only-in "regions.rkt" with-cleanup with-uninterruptible)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide adopted-bytes
         tr-tensor-free/checked
         collect-and-drain!
         swallow-and-count-failure
         finalizer-failures
         finalizer-diagnostics
         forget-gradient-adoptions!
         registering
         registering/release
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

(struct lifetime (state charge))

(define (fresh-lifetime)
  (lifetime (box 'live) (box #f)))

(define lifetimes (make-weak-hasheq))

(define (lifetime-of! key)
  (hash-ref! lifetimes key fresh-lifetime))

;; The tag flips even when the release raises: an attempted free consumes
;; the finalizer backstop, so a live-looking tag would invite use-after-free.
(define (release! h lt)
  (with-uninterruptible
    (with-cleanup
      (tr-tensor-free/unwrapped h)
      (race-point handle-released h)
      #:finally
      (set-cpointer-tag! h 'Tensor-freed)
      (when lt (set-box! (lifetime-state lt) 'freed)))))

;; a free killed before its release region leaves its dead thread as claimant
(define (claimable? held)
  (or (eq? held 'live) (and (thread? held) (thread-dead? held))))

(define (claim-lifetime! lt by)
  (cas-claim-when! (lifetime-state lt) claimable? by #:site 'lifetime))

(define (tr-tensor-free/checked h)
  (parameterize-break #f
    (define lt (hash-ref lifetimes h #f))
    (when (or (not lt) (claim-lifetime! lt (current-thread)))
      (when lt (unaccount-lifetime! h lt))
      (race-point free-unaccounted h)
      (release! h lt)
      (race-point free-released h))))

(struct failed (count messages))

(define failures (box (failed 0 '())))
(define capture-limit 8)

(define (finalizer-failures)
  (failed-count (unbox failures)))

(define entries (box 0))

(define (finalizer-diagnostics)
  (match-define (failed count messages) (unbox failures))
  (append (list (cons 'runs (finalizer-runs))
                (cons 'failures count)
                (cons 'messages (reverse messages))
                (cons 'ledger-entries (unbox entries)))
          (pressure-diagnostics)))

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
    (define message (describe-raised e))
    (void
     (cas-update! failures
                  (match-lambda
                    [(failed count messages)
                     (failed (add1 count)
                             (take-at-most capture-limit (cons message messages)))])))))

;; Total catch on purpose, not exn:fail?: any value escaping GC
;; finalization re-enters the error machinery and cascades.
(define (count-failed-run! e)
  (note-finalizer-run!)
  (record-failure! e))

(define ((swallow-and-count-failure release) t)
  (with-uninterruptible
    (with-handlers ([(lambda (_) #t) count-failed-run!])
      (release t)
      (note-finalizer-run!))))

(define (finalizer-for lt)
  (define release-claimed
    (swallow-and-count-failure
     (lambda (h)
       (race-point finalizer-releasing h)
       (unaccount-lifetime! h lt)
       (release! h lt))))
  (lambda (h)
    (with-uninterruptible
      (when (claim-lifetime! lt 'finalizing)
        (release-claimed h)))))

(define (register-tensor! h)
  (define lt (fresh-lifetime))
  (hash-set! lifetimes h lt)
  (register-finalizer h (finalizer-for lt))
  (race-point handle-registered h))

;; One uninterruptible region on the caller's OS thread: a kill or a break
;; waits for it, so no handle escapes without its finalizer.
(define (((registered register!) raw-fn) . args)
  (with-uninterruptible
    (define result (apply raw-fn args))
    (when result (register! result))
    result))

(define registering (registered register-tensor!))

(define registering/outputs
  (registered (lambda (handles)
                (for ([h (in-list handles)] #:when h)
                  (register-tensor! h)))))

(define (registering/release release)
  (define finalize (swallow-and-count-failure release))
  (registered (lambda (h) (register-finalizer h finalize))))

(struct allocation (phantom nbytes device account [adopted #:mutable] [adopter #:mutable]))

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

(define (native-placement t)
  (with-handlers ([exn:fail? (lambda (e)
                               (when (native-fault? e)
                                 (note-native-fault! "accounting a tensor"))
                               (values #f #f))])
    (define-values (nb-rc nbytes) (tr-tensor-nbytes/raw t))
    (define-values (dev-rc type index) (tr-tensor-device/raw t))
    (cond
      [(and (zero? nb-rc) (zero? dev-rc))
       (values nbytes (device type (if (eq? type 'cpu) 0 index)))]
      [else (values #f #f)])))

(define (account! t #:adopt [adopt #f])
  (define-values (nbytes dev) (native-placement t))
  (cond
    [dev (record-allocation! t nbytes dev #:adopt adopt)
         dev]
    [else #f]))

;; An `adopt` reads and writes the shadow, which lives under the ledger.
(define (record-allocation! key nbytes dev #:adopt [adopt #f])
  (define lt (lifetime-of! key))
  (define a (account-of dev))
  (define entry (allocation (make-phantom-bytes nbytes) nbytes dev a 0 #f))
  (define (charge!)
    (with-uninterruptible
      (set-box! (lifetime-charge lt) entry)
      (cas-add! entries 1)
      (race-point ledger-entry-added key)
      (note-accounted! a nbytes)
      (when adopt (adopt entry))))
  (if adopt (with-ledger (charge!)) (charge!)))

(define (unaccount! key)
  (define lt (hash-ref lifetimes key #f))
  (when lt (unaccount-lifetime! key lt)))

(define (unaccount-lifetime! key lt)
  (with-handlers ([exn:fail? void])
    (define charge (lifetime-charge lt))
    (define a (unbox charge))
    (when a
      (race-point unaccount-entry-read key)
      (with-uninterruptible
        (when (cas-claim! charge a #f)
          (cas-add! entries -1)
          (race-point entry-unaccounted key)
          (set-phantom-bytes! (allocation-phantom a) 0)
          (note-unaccounted! (allocation-account a) (allocation-nbytes a))
          ;; a reset zeroes the adoption of a gradient it dropped, under the ledger
          (when (positive? (allocation-adopted a))
            (with-ledger
              (race-point adoption-returning key)
              (when (positive? (allocation-adopted a))
                (unadopt! a)))))))))

(define (charge-of key)
  (define lt (hash-ref lifetimes key #f))
  (and lt (unbox (lifetime-charge lt))))

;; An in-place move (tr_tensor_to_) changes the device and byte count under
;; the same handle, so its ledger entry is replaced rather than added to. When
;; the new size cannot be read the old charge goes back: a stale entry still
;; presses on the collector, a missing one would not.
(define (reaccount! t)
  (define old (charge-of t))
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
  (define totals (make-hash))
  (for ([lt (in-list (hash-values lifetimes))])
    (define a (unbox (lifetime-charge lt)))
    (when a
      (hash-update! totals (allocation-device a)
                    (lambda (n) (+ n (allocation-nbytes a)))
                    0)))
  (sort-by-device (hash->list totals)))

;; `adopt` runs in the accounting's charge region, so no refresh reads the
;; handle's charge before the adoption moves it, and before the pressure
;; check, so the check sees the charge where it will stay.
(define (((accounted adopt) wrapped) . args)
  (define t (apply wrapped args))
  (race-point op-returned t)
  (when t
    (define dev (account! t #:adopt (and adopt (lambda (entry) (adopt args t entry)))))
    (race-point tensor-accounted t)
    (when dev
      (collect-under-pressure! dev)))
  t)

;; A gradient is storage the backward pass wrote before any handle pointed
;; at it, so the shadow's last refresh may charge it. The first handle taken
;; on a parameter's gradient since that refresh moves those bytes to the
;; ledger; a later one taken while any handle still holds an adoption, which
;; the ledger charges again as it does any second handle, moves nothing, even
;; across a refresh. At most one handle per parameter holds an adoption.
(struct adoption (generation entries))

(define no-adoption (adoption #f '()))

(define adopted-at (make-weak-hasheq))

(define (live-entries held)
  (filter weak-box-value (adoption-entries held)))

(define (held-by-a-live-adopter? held)
  (for/or ([b (in-list (adoption-entries held))])
    (define e (weak-box-value b))
    (and e (positive? (allocation-adopted e)))))

;; The caller holds the ledger.
(define (adopt-gradient! args t entry)
  (define param (car args))
  (race-point gradient-adopting t)
  (define now (shadow-generation))
  (define held (hash-ref adopted-at param no-adoption))
  (when (and (not (eqv? (adoption-generation held) now))
             (not (held-by-a-live-adopter? held)))
    (set-allocation-adopted!
     entry
     (note-adopted! (allocation-account entry) (allocation-nbytes entry)))
    (set-allocation-adopter! entry (make-weak-box param))
    (hash-set! adopted-at param
               (adoption now (cons (make-weak-box entry) (live-entries held))))))

(define (unadopt! a)
  (note-unadopted! (allocation-account a) (allocation-adopted a))
  (set-allocation-adopted! a 0)
  (define param (weak-box-value (allocation-adopter a)))
  (define held (and param (hash-ref adopted-at param #f)))
  (when held
    (hash-set! adopted-at param (adoption #f (live-entries held)))))

(define (adopted-bytes t)
  (with-ledger (allocation-adopted (charge-of t))))

;; The caller holds the ledger: the forgetting must share the lock with the
;; native reset that dropped the gradients.
(define (forget-gradient-adoptions! params)
  (race-point adoptions-forgotten params)
  (for ([param (in-list params)])
    (for ([held (in-list (live-entries (hash-ref adopted-at param no-adoption)))])
      (define entry (weak-box-value held))
      (when entry
        (set-allocation-adopted! entry 0)))
    (hash-remove! adopted-at param)))

;; The retry composes OUTSIDE the registration region: its collection waits
;; on the finalizer thread, which an uninterruptible region on main would
;; never let run.
(define (tensor-allocator raw-fn)
  ((accounted #f) ((oom-retry) (registering raw-fn))))

;; for tr_tensor_grad, whose first argument is the parameter
(define (tensor-allocator/gradient raw-fn)
  ((accounted adopt-gradient!) ((oom-retry) (registering raw-fn))))

;; No retry: re-running these after an OOM would repeat something the
;; first call already did — a draw from the global RNG stream, or an
;; in-place update of a tensor the caller handed in.
(define (tensor-allocator/no-retry raw-fn)
  ((accounted #f) (registering raw-fn)))

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
  (accounted-outputs ((oom-retry) (registering/outputs raw-fn))))

(define (tensor-allocator/outputs/no-retry raw-fn)
  (accounted-outputs (registering/outputs raw-fn)))

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

(define mem-trace-handle
  (and (getenv "RKTORCH_MEM_TRACE")
       (plumber-add-flush!
        (current-plumber)
        (lambda (_h)
          (with-handlers ([(lambda (_) #t) void])
            (eprintf "[rktorch mem] ~s\n" (finalizer-diagnostics)))))))
