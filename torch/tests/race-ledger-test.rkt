#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt" native-memory-limit native-memory-use zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/device-type.rkt" device)
           (only-in "../foreign/raw/collector.rkt"
                    call-as-the-collector collect-and-wait!)
           (only-in "../foreign/raw/memory.rkt" record-allocation! unaccount!)
           (only-in "../foreign/raw/pressure.rkt" since-check-of)
           (only-in "../foreign/structs.rkt" tensor-handle)
           (only-in "private/race-harness.rkt"
                    check-known-failure check-ledger gate-hook hooks join
                    labelled ledger-snapshot ledger-violations make-gate
                    make-watch
                    release-gate! settle! spawn stop! wait-for-arrival
                    worker-thread
                    watch-finalized watch-hook watch-max-holders
                    watch-release-count watch-released with-race-hook))

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
      (let* ([t (zeros 256 256)]
             [h (tensor-handle t)]
             [g (handle-gate 'free-unaccounted h)])
        (with-race-hook (hooks (watch-hook w) (gate-hook g))
          (define a (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a))
          (check-eq? (wait-for-arrival g) 'parked)
          (collect-and-wait!)
          (check-equal? (watch-release-count w h) 1
                        "a collection mid-free released the handle again")
          (release-gate! g)
          (join a)))
      (settle!))
    (check-equal? (watch-finalized w) 0
                  "the finalizer released an explicitly freed handle")
    (check-ledger base w))

  (test-case "an explicit free and the finalizer release a handle once (I3)"
    (for ([pool (in-list pools)])
      (explicit-free-against-finalizer pool)))

  (define (racing-frees pool)
    (settle!)
    (define w (make-watch))
    (define t (zeros 256 256))
    (define h (tensor-handle t))
    (define g (handle-gate 'free-unaccounted h))
    (with-race-hook (hooks (watch-hook w) (gate-hook g))
      (define a (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a))
      (check-eq? (wait-for-arrival g) 'parked)
      (join (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'b))
      (define committed (watch-release-count w h))
      (stop! a)
      committed))

  (test-case "two threads freeing one tensor reach the native release once (I3)"
    (for ([pool (in-list pools)])
      (check-known-failure
       "#266"
       (format "one release with ~a threads" (or pool 'ordinary))
       (= 1 (racing-frees pool)))))

  (define (double-unaccount pool)
    (define w (make-watch))
    (define key (box 'entry))
    (define dev (device 'cuda 13))
    (record-allocation! key 4096 dev)
    (define g
      (make-gate 'unaccount-entry-read
                 #:when (lambda (subject)
                          (and (eq? subject key) ((labelled 'a) subject)))))
    (with-race-hook (hooks (watch-hook w) (gate-hook g))
      (define a (spawn (lambda () (unaccount! key)) #:pool pool #:label 'a))
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

  (define (break-mid-free pool)
    (settle!)
    (define w (make-watch))
    (with-race-hook (watch-hook w)
      (let* ([t (zeros 256 256)]
             [g (handle-gate 'free-unaccounted (tensor-handle t))])
        (with-race-hook (hooks (watch-hook w) (gate-hook g))
          (define a (spawn (lambda () (tensor-free! t)) #:pool pool #:label 'a))
          (check-eq? (wait-for-arrival g) 'parked)
          (break-thread (worker-thread a))
          (release-gate! g)
          (check-exn exn:break? (lambda () (join a)))))
      (settle!))
    (check-equal? (watch-finalized w) 0
                  "a break abandoned the explicit free to the finalizer"))

  (test-case "a break during an explicit free waits until the release"
    (for ([pool (in-list pools)])
      (break-mid-free pool)))

  (define (killed-at point pool stop-by)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (define g (make-gate point #:when (labelled 'a)))
    (define owner (make-custodian))
    (with-race-hook (hooks (watch-hook w) (gate-hook g))
      (define a
        (parameterize ([current-custodian owner])
          (spawn (lambda () (void (zeros 256 256))) #:pool pool #:label 'a)))
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
    (with-race-hook (hooks (watch-hook w) (gate-hook g))
      (define a
        (spawn (lambda ()
                 (call-as-the-collector
                  (lambda ()
                    (semaphore-post inside)
                    (semaphore-wait leave))))
               #:pool pool #:label 'a))
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
      (define inside (make-semaphore 0))
      (define a
        (spawn (lambda ()
                 (call-as-the-collector
                  (lambda ()
                    (semaphore-post inside)
                    (semaphore-wait (make-semaphore 0)))))
               #:pool pool))
      (check-not-false (sync/timeout 30 inside))
      (stop! a)
      (check-eq? (join (spawn (lambda () (call-as-the-collector (lambda () 'ran)))
                              #:pool pool))
                 'ran)))

  (define (reset-against-add pool)
    (settle!)
    (define g (make-gate 'checks-resetting #:when (labelled 'a)))
    (with-race-hook (gate-hook g)
      (define a
        (spawn (lambda ()
                 (parameterize ([native-memory-limit mib])
                   (void (zeros 1024 1024))))
               #:pool pool #:label 'a))
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
    (define keys (list (box 'a) (box 'b)))
    (define g
      (make-gate 'account-created
                 #:when (lambda (subject)
                          (and (equal? subject dev) ((labelled 'a) subject)))))
    (with-race-hook (gate-hook g)
      (define a
        (spawn (lambda () (record-allocation! (car keys) 4096 dev))
               #:pool pool #:label 'a))
      (define created (wait-for-arrival g))
      (join (spawn (lambda () (record-allocation! (cadr keys) 4096 dev))
                   #:pool pool #:label 'b))
      (release-gate! g)
      (join a)
      (define live (cdr (or (assoc dev (native-memory-use)) (cons dev 0))))
      (for-each unaccount! keys)
      (check-equal? live 8192 "a racing creation lost a device's bytes")
      (check-ledger #f #f)
      (check-eq? created 'atomic "the account is created inside the ledger")))

  (test-case "two threads creating a device's first account lose no bytes"
    (first-account (device 'cuda 14) #f)
    (first-account (device 'cuda 15) 'own)))
