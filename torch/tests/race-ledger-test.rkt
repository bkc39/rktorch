#lang racket/base

(module+ test
  (require (only-in ffi/unsafe _fun)
           (only-in ffi/unsafe/atomic in-atomic-mode?)
           (only-in racket/dict dict-ref)
           rackunit
           (only-in "../foreign.rkt"
                    backward! grad native-memory-limit native-memory-use sum tensor?
                    zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free! to!)
           (only-in "../foreign/device-type.rkt" device)
           (only-in "../foreign/raw/cas.rkt" cas-add!)
           (only-in "../foreign/raw/collector.rkt" call-as-the-collector)
           (only-in "../foreign/raw/device-queries.rkt" allocator-reading)
           (only-in "../foreign/raw/memory.rkt"
                    native-memory-use/fold record-allocation! tensor-allocator unaccount!)
           (only-in "../foreign/raw/pressure.rkt"
                    live-bytes-by-device native-collect-margin refresh-shadows!
                    since-check-of)
           (only-in "../foreign/raw/syntax.rkt" _Tensor _Tensor/null define-torch)
           (only-in "../foreign/structs.rkt" tensor-handle)
           (only-in "private/race-harness.rkt"
                    check-ledger hooks join join-all labelled
                    ledger-snapshot ledger-violations make-gate make-watch
                    release-gate! seeded-generator settle! spawn stop! wait-for-arrival
                    watch-finalized watch-freed-count watch-hook
                    watch-max-holders watch-registered watch-release-count watch-released
                    while-inside with-parked with-race-hook worker-thread)
           (only-in "private/wedge.rkt" call-with-process-deadline))

  (define mib (* 1024 1024))
  (define cpu (device 'cpu 0))
  (define pools '(#f own))

  (define (handle-gate point h)
    (make-gate point #:when (lambda (subject) (eq? subject h))))

  (define (explicit-free-against-finalizer pool)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (with-race-hook (watch-hook w)
      (define t (zeros 256 256))
      (define g (handle-gate 'free-unaccounted (tensor-handle t)))
      (with-parked [a (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a)
                    #:at g #:watch w]
        (check-eq? (wait-for-arrival g) 'parked)
        (release-gate! g)
        (join a))
      (settle!))
    (check-equal? (watch-finalized w) 0
                  "the finalizer released an explicitly freed handle")
    (check-ledger base w))

  (test-case "an explicit free and the finalizer release a handle once (I3)"
    (for ([pool (in-list pools)])
      (explicit-free-against-finalizer pool)))

  (test-case "a tensor moved in place, then freed, is unaccounted once (I3)"
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (with-race-hook (watch-hook w)
      (define t (zeros 4 4))
      (to! t 'cpu 'float64)
      (tensor-free! t))
    (check-ledger base w))

  (define freeing-loser? #rx"tensor-free!: the tensor is being freed or was freed")

  (define (racing-frees point pool)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (with-race-hook (watch-hook w)
      (define t (zeros 256 256))
      (define h (tensor-handle t))
      (define g (handle-gate point h))
      (define (free-by label)
        (spawn (lambda () (tensor-free! t)) #:pool pool #:label label))
      (with-parked [a (free-by 'a) #:at g #:watch w]
        (check-eq? (wait-for-arrival g) 'parked)
        (check-exn freeing-loser? (lambda () (join (free-by 'b))))
        (settle!)
        (release-gate! g)
        (join a))
      (settle!)
      (check-equal? (watch-release-count w h) 1
                    (format "releases of a handle two ~a threads freed, one parked at ~a"
                            (or pool 'ordinary) point))
      (check-equal? (watch-freed-count w h) 1))
    (check-ledger base w))

  (test-case "two threads freeing one tensor reach the native release once (I3)"
    (for* ([point (in-list '(free-unaccounted free-taken))]
           [pool (in-list pools)])
      (racing-frees point pool)))

  (define (double-unaccount pool)
    (define w (make-watch))
    (define key (box 'entry))
    (define dev (device 'cuda 13))
    (record-allocation! key 4096 dev)
    (define g
      (make-gate 'unaccount-entry-read
                 #:when (lambda (subject)
                          (and (eq? subject key) ((labelled 'a) subject)))))
    (with-parked [a (spawn (lambda () (unaccount! key)) #:pool pool #:label 'a)
                  #:at g #:watch w]
      (define read (wait-for-arrival g))
      (join (spawn (lambda () (unaccount! key)) #:pool pool #:label 'b))
      (define witness (box 'witness))
      (record-allocation! witness 4096 dev)
      (release-gate! g)
      (join a)
      (define violations (ledger-violations #:watch w))
      (unaccount! witness)
      (check-equal? violations '())
      (check-eq? read 'parked "the entry is read before the CAS that takes it")))

  (test-case "two threads unaccounting one entry subtract it once (I3)"
    (for ([pool (in-list pools)])
      (double-unaccount pool)))

  (define (break-mid-free point pool)
    (settle!)
    (define w (make-watch))
    (with-race-hook (watch-hook w)
      (define t (zeros 256 256))
      (define h (tensor-handle t))
      (define g (handle-gate point h))
      (with-parked [a (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a)
                    #:at g #:watch w]
        (check-eq? (wait-for-arrival g) 'parked)
        (break-thread (worker-thread a))
        (release-gate! g)
        (check-exn exn:break? (lambda () (join a)))
        (check-equal? (watch-freed-count w h) 1
                      "a break cut the explicit free off before its release"))
      (settle!))
    (check-equal? (watch-finalized w) 0
                  "a break abandoned the explicit free to the finalizer"))

  (test-case "a break during an explicit free waits until the release"
    (for* ([point (in-list '(free-taken free-unaccounted))]
           [pool (in-list pools)])
      (break-mid-free point pool)))

  (define (stop-mid-free! point pool stop-by)
    (define owner (make-custodian))
    (define t (zeros 256 256))
    (define h (tensor-handle t))
    (define g (handle-gate point h))
    (define parked (make-watch))
    (with-parked [a (parameterize ([current-custodian owner])
                      (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a))
                  #:at g #:watch parked]
      (check-eq? (wait-for-arrival g) 'parked)
      (case stop-by
        [(kill) (stop! a)]
        [(custodian) (custodian-shutdown-all owner)])
      (check-true (tensor? t) "a killed free flipped the tag")
      (check-exn freeing-loser? (lambda () (tensor-free! t)))
      (check-equal? (watch-freed-count parked h) 0
                    "a native release ran while the killed free was parked")))

  (define (killed-mid-free point pool stop-by)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (with-race-hook (watch-hook w)
      (stop-mid-free! point pool stop-by)
      (settle!))
    (check-equal? (watch-finalized w) 1
                  (format "the finalizer of a free killed at ~a (~a, ~a)"
                          point (or pool 'ordinary) stop-by))
    (check-ledger base w))

  (test-case "a free killed mid-way leaves the handle to the finalizer"
    (for* ([point (in-list '(free-taken free-unaccounted))]
           [pool (in-list pools)]
           [stop-by (in-list '(kill custodian))])
      (killed-mid-free point pool stop-by)))

  (define (killed-after-unaccounting pool)
    (define t (zeros 256 256))
    (define g (handle-gate 'free-unaccounted (tensor-handle t)))
    (define (live) (dict-ref (native-memory-use) cpu 0))
    (define held (live))
    (with-parked [a (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a)
                  #:at g]
      (check-eq? (wait-for-arrival g) 'parked)
      (stop! a))
    (check-equal? (- held (live)) (* 256 256 4)
                  "a free killed after it unaccounted left its bytes on the ledger")
    (check-true (tensor? t) "a free killed before its release flipped the tag"))

  (test-case "a free killed after it unaccounted leaves a live tensor off the ledger"
    (for ([pool (in-list pools)])
      (settle!)
      (define base (ledger-snapshot))
      (define w (make-watch))
      (with-race-hook (watch-hook w)
        (killed-after-unaccounting pool)
        (settle!))
      (check-equal? (watch-finalized w) 1)
      (check-ledger base w)))

  (define (killed-at point pool stop-by)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (define g (make-gate point #:when (labelled 'a)))
    (define owner (make-custodian))
    (with-parked [a (parameterize ([current-custodian owner])
                      (spawn (lambda () (void (zeros 256 256)))
                             #:pool pool #:label 'a))
                  #:at g #:watch w]
      (check-eq? (wait-for-arrival g) 'parked)
      (case stop-by
        [(kill) (stop! a)]
        [(custodian) (custodian-shutdown-all owner)])
      (settle!)
      (check-equal? (watch-released w) 1 "the finalizer never released it"))
    (check-ledger base w))

  (test-case "a thread killed mid-allocation leaks nothing"
    (for* ([point (in-list '(op-returned gate-read))]
           [pool (in-list pools)]
           [stop-by (in-list '(kill custodian))])
      (killed-at point pool stop-by)))

  (define (rival-collectors pool)
    (define w (make-watch))
    (define g (make-gate 'collector-claiming #:when (labelled 'a)))
    (define inside (make-semaphore 0))
    (define leave (make-semaphore 0))
    (with-parked [a (spawn (lambda ()
                             (call-as-the-collector
                              (lambda ()
                                (semaphore-post inside)
                                (semaphore-wait leave))))
                           #:pool pool #:label 'a)
                  #:at g #:watch w]
      (define claimed (wait-for-arrival g))
      (define b
        (spawn (lambda ()
                 (call-as-the-collector
                  (lambda ()
                    (release-gate! g)
                    (sync/timeout 1 inside))))
               #:pool pool #:label 'b))
      (join b)
      (release-gate! g)
      (semaphore-post leave)
      (join a)
      (check-ledger #f w)
      (check-equal? (watch-max-holders w) 1)
      (check-eq? claimed 'parked "the claim is read before the CAS that takes it")))

  (test-case "two threads claiming the collector hold it one at a time (I2)"
    (for ([pool (in-list pools)])
      (rival-collectors pool)))

  (test-case "a collector killed mid-collection leaves the claim free"
    (for ([pool (in-list pools)])
      (define (kill-then-claim a)
        (stop! a)
        (join (spawn (lambda () (call-as-the-collector (lambda () 'ran)))
                     #:pool pool)))
      (check-eq? (while-inside call-as-the-collector kill-then-claim #:pool pool)
                 'ran)))

  (define (reset-against-add pool)
    (settle!)
    (define g (make-gate 'checks-resetting #:when (labelled 'a)))
    (with-parked [a (spawn (lambda ()
                             (parameterize ([native-memory-limit mib])
                               (void (zeros 1024 1024))))
                           #:pool pool #:label 'a)
                  #:at g]
      (check-eq? (wait-for-arrival g) 'parked)
      (define before (since-check-of cpu))
      (define held (zeros 256 256))
      (define added (- (since-check-of cpu) before))
      (release-gate! g)
      (join a)
      (check-equal? added (* 256 256 4))
      (check-true (>= (since-check-of cpu) added)
                  (format "a reset keeps the bytes added during its collection (~a)"
                          (or pool 'ordinary)))
      (tensor-free! held)))

  (test-case "a counter reset racing an allocation"
    (for ([pool (in-list pools)])
      (reset-against-add pool)))

  (define (first-account dev pool)
    (define a-key (box 'a))
    (define b-key (box 'b))
    (define g
      (make-gate 'account-created
                 #:when (lambda (subject)
                          (and (equal? subject dev) ((labelled 'a) subject)))))
    (with-parked [a (spawn (lambda () (record-allocation! a-key 4096 dev))
                           #:pool pool #:label 'a)
                  #:at g]
      (define created (wait-for-arrival g))
      (join (spawn (lambda () (record-allocation! b-key 4096 dev))
                   #:pool pool #:label 'b))
      (release-gate! g)
      (join a)
      (define live (dict-ref (native-memory-use) dev 0))
      (unaccount! a-key)
      (unaccount! b-key)
      (check-equal? live 8192 "a racing creation lost a device's bytes")
      (check-ledger #f #f)
      (check-eq? created 'parked "the account is made before the CAS that inserts it")))

  (test-case "two threads creating a device's first account lose no bytes"
    (first-account (device 'cuda 14) #f)
    (first-account (device 'cuda 15) 'own))

  (define (spin-for ms)
    (define until (+ (current-inexact-milliseconds) ms))
    (let spin ()
      (when (< (current-inexact-milliseconds) until)
        (spin))))

  (define (await-box b #:seconds [seconds 30])
    (define deadline (+ (current-inexact-milliseconds) (* 1000 seconds)))
    (let poll ()
      (cond
        [(unbox b) (unbox b)]
        [(> (current-inexact-milliseconds) deadline) #f]
        [else (sleep 0.001) (poll)])))

  ;; The count is bumped inside the raw call, so the registration region
  ;; covers it: a handle the native call made but no finalizer guards shows
  ;; as a count above the registrations. The spin widens the window a kill
  ;; must miss.
  (define-torch neg/bare (_fun _Tensor -> _Tensor/null) #:c-id tr_neg)

  (define trials 200)

  (define (stopped-mid-registration pool stop-by)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (define made (box 0))
    (define x (zeros 4 4))
    (define op
      (tensor-allocator (lambda (t) (begin0 (neg/bare t) (cas-add! made 1) (spin-for 0.05)))))
    (define rng (seeded-generator 296 (list pool stop-by)))
    (with-race-hook (watch-hook w)
      (for ([_ (in-range trials)])
        (define started (box #f))
        (define a
          (spawn (lambda ()
                   (let loop ()
                     (void (op x))
                     (set-box! started #t)
                     (loop)))
                 #:pool pool))
        (await-box started)
        (sleep (* 0.002 (random rng)))
        (case stop-by
          [(kill) (stop! a)]
          [(break)
           (break-thread (worker-thread a))
           (check-not-false (sync/timeout 30 (thread-dead-evt (worker-thread a)))
                            "a broken worker kept running")]))
      (settle!))
    (tensor-free! x)
    (define where (format "~a, ~a" (or pool 'ordinary) stop-by))
    (check-equal? (watch-registered w) (unbox made)
                  (format "a handle escaped its registration (~a)" where))
    (check-equal? (watch-finalized w) (watch-registered w)
                  (format "a registered handle was not released once (~a)" where))
    (check-ledger base w))

  (test-case "a thread stopped mid-registration leaves no handle unguarded"
    (for* ([stop-by (in-list '(kill break))]
           [pool (in-list pools)])
      (stopped-mid-registration pool stop-by)))

  (test-case "eight parallel threads accounting and unaccounting keep the counter at the fold"
    (define dev (device 'cuda 16))
    (join-all
     (for/list ([i (in-range 8)])
       (spawn (lambda ()
                (for ([_ (in-range 10000)])
                  (define key (box i))
                  (record-allocation! key 4096 dev)
                  (unaccount! key)))
              #:pool 'own)))
    (check-equal? (dict-ref (live-bytes-by-device) dev #f) 0)
    (check-equal? (dict-ref (native-memory-use/fold) dev 0) 0)
    (check-ledger #f #f))

  ;; The finalizer takes the ledger only to give an adoption back, so the
  ;; dropped gradient must hold one: the shadow charges the gradient first.
  (define (adopted-gradient-dropped!)
    (define p (zeros 256 256 #:requires-grad? #t))
    (backward! (sum p))
    (refresh-shadows!)
    (void (grad p))
    p)

  (define (finalizer-against-lock-holder)
    (settle!)
    (define w (make-watch))
    (define held (box #f))
    (define y (zeros 4 4 #:requires-grad? #t))
    (backward! (sum y))
    (define (hold-the-ledger name _subject)
      (when (and (eq? name 'gradient-adopting) ((labelled 'holder) #f))
        (set-box! held #t)
        (spin-for 50)))
    (parameterize ([allocator-reading (lambda (_dev) (* 64 mib))]
                   [native-collect-margin (* 1024 1024 mib)])
      (define p (adopted-gradient-dropped!))
      (with-race-hook (hooks (watch-hook w) hold-the-ledger)
        (define holder (spawn (lambda () (void (grad y))) #:pool 'own #:label 'holder))
        (check-true (await-box held) "the holder never reached the ledger")
        (collect-garbage)
        (sleep 0)
        (join holder)
        (settle!))
      (check-true (positive? (watch-finalized w)) "no finalizer ran against the holder")
      (check-ledger #f w)
      p))

  (test-case "a finalizer waits out a parallel thread holding the ledger"
    (void (call-with-process-deadline 30 finalizer-against-lock-holder)))

  ;; #423: no kill separates the native release from the tag flip.
  (define (killed-mid-release pool)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (define t (zeros 256 256))
    (define h (tensor-handle t))
    (define arrived (box #f))
    (define (spin-at-release name subject)
      (when (and (eq? name 'handle-released) (eq? subject h))
        (set-box! arrived (if (in-atomic-mode?) 'in-region 'outside))
        (spin-for 50)))
    (with-race-hook (hooks (watch-hook w) spin-at-release)
      (define a (spawn (lambda () (tensor-free! t)) #:pool pool))
      (define where (await-box arrived))
      (stop! a)
      (check-false (tensor? t) "a kill left a released handle tagged live")
      (check-eq? where 'in-region
                 "the release and the tag flip share one uninterruptible region")
      (check-equal? (watch-release-count w h) 1)
      (settle!))
    (check-equal? (watch-finalized w) 0 "the finalizer released a freed handle again")
    (check-ledger base w))

  (test-case "a thread killed between the release and the tag flip leaves the handle freed"
    (for ([pool (in-list pools)])
      (killed-mid-release pool))))
