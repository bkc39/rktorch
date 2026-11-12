#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe
                  _enum _fun _int _int64 _list _ptr _void set-cpointer-tag!)
         (only-in racket/list filter-map remove-duplicates take)
         (only-in racket/match match-define match-lambda)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../runtime/reaper.rkt" reap-when-dead! reaped)
         (only-in "../device-type.rkt" device device-index device-type)
         (only-in "accounts.rkt"
                  count-checked! count-live! ledger-entries tally-of with-ledger)
         (only-in "cas.rkt" cas-claim! cas-claim-when! cas-update!)
         (only-in "collector.rkt"
                  finalizer-runs
                  note-finalizer-run!)
         (only-in "fault.rkt" native-fault? note-native-fault! with-fault-handlers)
         (only-in "groups.rkt" fresh-group lower-group! settle!)
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
                  note-adopted!
                  note-unadopted!
                  pressure-diagnostics
                  shadow-generation
                  unaccounted-bytes-by-device)
         (only-in "race-points.rkt" race-point)
         (only-in "regions.rkt" with-cleanup with-uninterruptible)
         (only-in "syntax.rkt" _Tensor define-torch)
         (only-in "tallies.rkt" flush-tally! tally-charge! tally-group-add! tally-op!))

(provide account!
         adopted-bytes
         tr-tensor-free/checked
         free-handles!
         collect-and-drain!
         swallow-and-count-failure
         finalizer-failures
         finalizer-diagnostics
         forget-gradient-adoptions!
         adopt-gradient!
         registered
         register-tensors!
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
         tr-tensor-free/unwrapped)

(define-torch tr-tensor-free/unwrapped
  (_fun _Tensor -> _void) #:c-id tr_tensor_free)

(define-torch tr-tensors-free/unwrapped
  (_fun (hs : (_list i _Tensor)) (_int64 = (length hs)) -> _void)
  #:c-id tr_tensors_free)

(struct lifetime (state charge))

(define (fresh-lifetime) (lifetime (box 'live) (box #f)))

(define lifetimes (make-weak-hasheq))

(define (lifetime-of! key)
  (hash-ref! lifetimes key fresh-lifetime))

;; The tag flips even when the release raises: an attempted free consumes
;; the finalizer backstop, so a live-looking tag would invite use-after-free.
(define (mark-freed! h lt)
  (set-cpointer-tag! h 'Tensor-freed)
  (when lt (set-box! (lifetime-state lt) 'freed)))

(define (release! h lt)
  (with-uninterruptible
    (with-cleanup
      (tr-tensor-free/unwrapped h)
      (race-point handle-released h)
      #:finally
      (mark-freed! h lt))))

(define (release-all! hs lts)
  (with-uninterruptible
    (with-cleanup
      (tr-tensors-free/unwrapped hs)
      (for-each (lambda (h) (race-point handle-released h)) hs)
      #:finally
      (for-each mark-freed! hs lts))))

;; a free killed before its release region leaves its dead thread as claimant
(define (claimable? held)
  (or (eq? held 'live) (and (thread? held) (thread-dead? held))))

(define (claim-lifetime! lt by)
  (cas-claim-when! (lifetime-state lt) claimable? by #:site 'lifetime))

(define (orphaned? held)
  (or (eq? held 'live) (thread? held)))

(define (claim-dead-lifetime! lt)
  (cas-claim-when! (lifetime-state lt) orphaned? 'finalizing #:site 'lifetime))

(define (tr-tensor-free/checked h)
  (parameterize-break #f
    (define lt (hash-ref lifetimes h #f))
    (when (or (not lt) (claim-lifetime! lt (current-thread)))
      (when lt (unaccount-lifetime! h lt))
      (race-point free-unaccounted h)
      (release! h lt)
      (race-point free-released h))))

;; Each handle is claimed as `tr-tensor-free/checked` claims it; only the
;; handles this call claimed reach the one native release.
(define (free-handles! hs)
  (parameterize-break #f
    (define claimed
      (filter-map (lambda (h)
                    (define lt (hash-ref lifetimes h #f))
                    (and (or (not lt) (claim-lifetime! lt (current-thread)))
                         (cons h lt)))
                  hs))
    (define held (map car claimed))
    (define lts (map cdr claimed))
    (for ([h (in-list held)] [lt (in-list lts)])
      (when lt (unaccount-lifetime! h lt))
      (race-point free-unaccounted h))
    (unless (null? held)
      (release-all! held lts))
    (for-each (lambda (h) (race-point free-released h)) held)
    (race-point frees-released/batched held)))

(struct failed (count messages))

(define failures (box (failed 0 '())))
(define capture-limit 8)

(define (finalizer-failures)
  (failed-count (unbox failures)))

(define (finalizer-diagnostics)
  (match-define (failed count messages) (unbox failures))
  (append (list (cons 'runs (finalizer-runs))
                (cons 'failures count)
                (cons 'messages (reverse messages))
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
  (take xs (min n (length xs))))

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

(define-syntax-parse-rule (counting-failures body:expr ...+)
  (with-uninterruptible
    (with-fault-handlers ([(lambda (_) #t) count-failed-run!])
      body ...
      (note-finalizer-run!))))

(define ((swallow-and-count-failure release) t)
  (counting-failures (release t)))

(define (release-dead-handle! h)
  (define lt (hash-ref lifetimes h #f))
  (cond
    [(not lt) (count-failed-run! "a dead handle with no lifetime")]
    [(claim-dead-lifetime! lt)
     (counting-failures
      (race-point finalizer-releasing h)
      (unaccount-lifetime! h lt)
      (release! h lt))]
    [else (void)]))

(define tensor-reaped (reaped 'tensor release-dead-handle! #:after-batch flush-tally!))

(define (register-tensor! h)
  (hash-set! lifetimes h (fresh-lifetime))
  (reap-when-dead! tensor-reaped h)
  (race-point handle-registered h))

;; One uninterruptible region on the caller's OS thread: a kill or a break
;; waits for it, so no handle escapes without its finalizer.
(define (((registered register!) raw-fn) . args)
  (with-uninterruptible
    (define result (apply raw-fn args))
    (when result (register! result))
    result))

(define registering (registered register-tensor!))

(define (register-tensors! handles)
  (for ([h (in-list handles)] #:when h)
    (register-tensor! h)))

(define registering/outputs (registered register-tensors!))

(define (registering/release release)
  (define (release-generator! h)
    (race-point generator-releasing h)
    (release h))
  (define kind (reaped 'generator (swallow-and-count-failure release-generator!)))
  (registered (lambda (h) (reap-when-dead! kind h))))

(struct allocation (group nbytes device account [adopted #:mutable] [adopter #:mutable]))

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
  (with-fault-handlers ([exn:fail? (lambda (e)
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

;; An `adopt` reads and writes the shadow, which lives under the ledger. The
;; count and the live bytes go up before the entry is published, so an
;; unaccount racing the publication never takes either below what it added;
;; a handle already charged keeps its charge, and the loser takes both back.
(define (record-allocation! key nbytes dev #:adopt [adopt #f])
  (define lt (lifetime-of! key))
  (define a (account-of dev))
  (define (charge! p)
    (with-uninterruptible
      (define g (if p (tally-charge! p nbytes) (fresh-group nbytes)))
      (define entry (allocation g nbytes dev a 0 #f))
      (count-live! a p nbytes 1)
      (race-point entry-counted key)
      (cond
        [(cas-claim! (lifetime-charge lt) #f entry)
         (race-point ledger-entry-added key)
         (count-checked! a p nbytes)
         (unless p (settle! g))
         (when adopt (adopt entry))]
        [else
         (count-live! a p (- nbytes) -1)
         (when p (tally-group-add! p g (- nbytes)))])
      (when p (tally-op! p nbytes))))
  (if adopt (with-ledger (charge! #f)) (charge! (tally-of a))))

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
          (release-entry! key a))))))

(define (release-entry! key a)
  (match-define (allocation group nbytes _ account adopted _) a)
  (define p (and (zero? adopted) (tally-of account)))
  (count-live! account p (if p (- nbytes) 0) -1)
  (race-point entry-unaccounted key)
  ;; an adoption's bytes leave the ledger and return to the shadow in
  ;; one section: a refresh between them would charge them twice. A
  ;; reset zeroes the adoption of a gradient it dropped, under the ledger.
  (cond
    [p (tally-group-add! p group (- nbytes))
       (tally-op! p nbytes)]
    [else
     (lower-group! group nbytes)
     (with-ledger
       (count-live! account #f (- nbytes) 0)
       (race-point live-returned key)
       (race-point adoption-returning key)
       (when (positive? (allocation-adopted a))
         (unadopt! a)))]))

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

(define mem-trace-handle
  (and (getenv "RKTORCH_MEM_TRACE")
       (plumber-add-flush!
        (current-plumber)
        (lambda (_h)
          (with-handlers ([(lambda (_) #t) void])
            (eprintf "[rktorch mem] ~s\n" (finalizer-diagnostics)))))))
