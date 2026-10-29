#lang racket/base

(module+ test
  (require (only-in racket/dict dict-ref)
           rackunit
           (only-in "../foreign.rkt"
                    native-memory-limit native-memory-use tensor? zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free! to!)
           (only-in "../foreign/device-type.rkt" device)
           (only-in "../foreign/raw/collector.rkt" call-as-the-collector)
           (only-in "../foreign/raw/memory.rkt" record-allocation! unaccount!)
           (only-in "../foreign/raw/pressure.rkt" since-check-of)
           (only-in "../foreign/structs.rkt" tensor-handle)
           (only-in "private/race-harness.rkt"
                    check-known-failure check-ledger join labelled
                    ledger-snapshot ledger-violations make-gate make-watch
                    release-gate! settle! spawn stop! wait-for-arrival
                    watch-finalized watch-freed-count watch-hook
                    watch-max-holders watch-release-count watch-released
                    while-inside with-parked with-race-hook worker-thread))

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
      (check-eq? read 'atomic "the entry is read and removed in one section")))

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
    (define g (handle-gate point (tensor-handle t)))
    (with-parked [a (parameterize ([current-custodian owner])
                      (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a))
                  #:at g]
      (check-eq? (wait-for-arrival g) 'parked)
      (case stop-by
        [(kill) (stop! a)]
        [(custodian) (custodian-shutdown-all owner)])
      (check-true (tensor? t) "a killed free flipped the tag")
      (check-exn freeing-loser? (lambda () (tensor-free! t)))))

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
      (check-eq? claimed 'atomic "the claim is decided in one atomic section")))

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
      (check-known-failure
       "#168 stage 2"
       (format "a reset keeps the bytes added during its collection (~a)"
               (or pool 'ordinary))
       (>= (since-check-of cpu) added))
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
      (check-eq? created 'atomic "the account is created inside the ledger")))

  (test-case "two threads creating a device's first account lose no bytes"
    (first-account (device 'cuda 14) #f)
    (first-account (device 'cuda 15) 'own)))
