#lang racket/base

(require (for-syntax racket/base
                     ;; whole-module require on purpose
                     syntax/parse/pre)
         (only-in ffi/unsafe
                  _double _enum _fun _int _int64 _ptr _void register-finalizer
                  set-cpointer-tag!)
         (only-in ffi/unsafe/atomic call-as-uninterruptible)
         (only-in racket/list remove-duplicates)
         (only-in "../device-type.rkt" device device-index device-type)
         (only-in "cas.rkt" cas-claim! cas-update!)
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
                  call-with-ledger
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
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide adopted-bytes
         tr-tensor-free/checked
         collect-and-drain!
         swallow-and-count-failure
         finalizer-failures
         finalizer-diagnostics
         forget-gradient-adoptions!
         handle-allocator
         registering
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

;; A handle's lifetime: `state` decides which of an explicit free and the
;; finalizer releases it ('live -> the claimant by CAS, then 'freed), and
;; `charge` holds its ledger entry, which an unaccounting takes by CAS to #f
;; so that only one subtracts it. The finalizer closes over the lifetime, so
;; it never needs the weak table's entry.
(struct lifetime (state charge))

(define (fresh-lifetime)
  (lifetime (box 'live) (box #f)))

(define lifetimes (make-weak-hasheq))

(define (lifetime-of! key)
  (hash-ref! lifetimes key fresh-lifetime))

;; One region holds the native release, the 'Tensor-freed tag and the
;; 'freed state, so a kill cannot leave a released handle tagged live (#423).
;; The tag flips even when the release raises: an attempted free consumes
;; the finalizer backstop, so a live-looking tag would invite use-after-free.
(define (release! h lt)
  (call-as-uninterruptible
   (lambda ()
     (dynamic-wind
      void
      (lambda ()
        (tr-tensor-free/unwrapped h)
        (race-point handle-released h))
      (lambda ()
        (set-cpointer-tag! h 'Tensor-freed)
        (when lt (set-box! (lifetime-state lt) 'freed)))))))

;; A claim names its claimant. A free killed before its release region
;; holds the claim with a dead thread, and the finalizer may take it over:
;; the dead free will neither release the handle nor unaccount it.
(define (claim-lifetime! lt by)
  (define state (lifetime-state lt))
  (let retry ()
    (define held (unbox state))
    (cond
      [(not (or (eq? held 'live) (and (thread? held) (thread-dead? held)))) #f]
      [(box-cas! state held by) #t]
      [else (retry)])))

;; The explicit free: the claim decides it against the finalizer and any
;; other free of the handle, and the loser returns. A handle no allocator
;; registered has no lifetime and nothing to race.
(define (tr-tensor-free/checked h)
  (parameterize-break #f
    (define lt (hash-ref lifetimes h #f))
    (when (or (not lt) (claim-lifetime! lt (current-thread)))
      (when lt (unaccount-lifetime! h lt))
      (race-point free-unaccounted h)
      (release! h lt)
      (race-point free-released h))))

(define failures (box (cons 0 '())))
(define capture-limit 8)

(define (finalizer-failures)
  (car (unbox failures)))

(define (ledger-entries)
  (for/sum ([lt (in-list (hash-values lifetimes))])
    (if (unbox (lifetime-charge lt)) 1 0)))

(define (finalizer-diagnostics)
  (define seen (unbox failures))
  (append (list (cons 'runs (finalizer-runs))
                (cons 'failures (car seen))
                (cons 'messages (reverse (cdr seen)))
                (cons 'ledger-entries (ledger-entries)))
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
                  (lambda (seen)
                    (cons (add1 (car seen))
                          (take-at-most capture-limit (cons message (cdr seen)))))))))

;; Total catch on purpose, not exn:fail?: any value escaping GC
;; finalization re-enters the error machinery and cascades (#38).
;; A plain `register-finalizer` runs its finalizer outside atomic mode, so
;; the region keeps another green thread from running in the middle of one,
;; and keeps a drain from seeing a counted run that has not released yet.
(define ((swallow-and-count-failure release) t)
  (call-as-uninterruptible
   (lambda ()
     (with-handlers ([(lambda (_) #t) record-failure!])
       (note-finalizer-run!)
       (release t)))))

;; The finalizer counts a run only when its claim wins: a handle an explicit
;; free took is not its to release.
(define (finalizer-for lt)
  (define release-claimed
    (swallow-and-count-failure
     (lambda (h)
       (race-point finalizer-releasing h)
       (unaccount-lifetime! h lt)
       (release! h lt))))
  (lambda (h)
    (call-as-uninterruptible
     (lambda ()
       (when (claim-lifetime! lt 'finalizing)
         (release-claimed h))))))

(define (register-tensor! h)
  (define lt (fresh-lifetime))
  (hash-set! lifetimes h lt)
  (register-finalizer h (finalizer-for lt))
  (race-point handle-registered h))

;; The raw call and the registration of each handle it answers share one
;; uninterruptible region: it stays on the caller's OS thread, and a kill or
;; a break waits for it, so no handle escapes without its finalizer.
(define ((registering raw-fn) . args)
  (call-as-uninterruptible
   (lambda ()
     (define h (apply raw-fn args))
     (when h (register-tensor! h))
     h)))

(define ((registering/outputs raw-fn) . args)
  (call-as-uninterruptible
   (lambda ()
     (define handles (apply raw-fn args))
     (when handles
       (for ([h (in-list handles)] #:when h)
         (register-tensor! h)))
     handles)))

;; For handles with no bytes to charge, a generator: the guarded finalizer
;; alone releases it.
(define (((handle-allocator release) raw-fn) . args)
  (define finalize (swallow-and-count-failure release))
  (call-as-uninterruptible
   (lambda ()
     (define h (apply raw-fn args))
     (when h (register-finalizer h finalize))
     h)))

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

;; The charge and the counters move in one uninterruptible region, so no kill
;; separates them. An `adopt` takes the ledger first: it reads and writes the
;; shadow, which lives under the ledger.
(define (record-allocation! key nbytes dev #:adopt [adopt #f])
  (define lt (lifetime-of! key))
  (define a (account-of dev))
  (define entry (allocation (make-phantom-bytes nbytes) nbytes dev a 0 #f))
  (define (charge!)
    (call-as-uninterruptible
     (lambda ()
       (set-box! (lifetime-charge lt) entry)
       (race-point ledger-entry-added key)
       (note-accounted! a nbytes)
       (when adopt (adopt entry)))))
  (if adopt (call-with-ledger charge!) (charge!)))

(define (unaccount! key)
  (define lt (hash-ref lifetimes key #f))
  (when lt (unaccount-lifetime! key lt)))

(define (unaccount-lifetime! key lt)
  (with-handlers ([exn:fail? void])
    (define charge (lifetime-charge lt))
    (define a (unbox charge))
    (when a
      (race-point unaccount-entry-read key)
      (call-as-uninterruptible
       (lambda ()
         (when (cas-claim! charge a #f)
           (race-point entry-unaccounted key)
           (set-phantom-bytes! (allocation-phantom a) 0)
           (note-unaccounted! (allocation-account a) (allocation-nbytes a))
           ;; a positive adoption means the parameter still holds the gradient
           ;; this handle took over: a reset zeroes it first, under the ledger
           (when (positive? (allocation-adopted a))
             (call-with-ledger
              (lambda ()
                (when (positive? (allocation-adopted a))
                  (unadopt! a)))))))))))

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
  (call-with-ledger (lambda () (allocation-adopted (charge-of t)))))

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

;; No I/O from inside a finalizer -- that runs uninterruptibly, where writing
;; to a port can block and trip an internal error.  Accumulate, dump here.
(define mem-trace-handle
  (and (getenv "RKTORCH_MEM_TRACE")
       (plumber-add-flush!
        (current-plumber)
        (lambda (_h)
          (with-handlers ([(lambda (_) #t) void])
            (eprintf "[rktorch mem] ~s\n" (finalizer-diagnostics)))))))
