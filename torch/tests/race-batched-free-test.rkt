#lang racket/base

(module+ test
  (require (only-in ffi/unsafe cpointer-has-tag?)
           rackunit
           (only-in "../foreign.rkt" tensor? zeros)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/structs.rkt" tensor-handle tensors-free!)
           (only-in "private/race-harness.rkt"
                    check-ledger hooks join ledger-snapshot make-gate make-watch
                    release-gate! settle! spawn stop! wait-for-arrival
                    watch-finalized watch-freed-count watch-hook watch-release-count
                    with-parked with-race-hook))

  (define pools '(#f own))

  (define (handle-gate point h)
    (make-gate point #:when (lambda (subject) (eq? subject h))))

  ;; A finalizer reaching a handle a free already released would release it
  ;; natively a second time; refusing there turns that into a counted failure.
  (define (refuse-released-finalizer name subject)
    (when (and (eq? name 'finalizer-releasing)
               (cpointer-has-tag? subject 'Tensor-freed))
      (raise 'released-twice)))

  (define (batch-free-and-drop! pool)
    (define kept (zeros 256 256))
    (define dropped (zeros 256 256))
    (join (spawn (lambda () (tensors-free! (list kept (zeros 256 256))))
                 #:pool pool #:label 'batch))
    (check-false (tensor? kept) "the batched free left a live tag")
    (void dropped))

  (test-case "a batched free and the finalizer release each handle once"
    (for ([pool (in-list pools)])
      (settle!)
      (define base (ledger-snapshot))
      (define w (make-watch))
      (with-race-hook (hooks (watch-hook w) refuse-released-finalizer)
        (batch-free-and-drop! pool)
        (settle!))
      (check-equal? (watch-finalized w) 1
                    (format "finalizers run after a batched free (~a)" (or pool 'ordinary)))
      (check-ledger base w)))

  (define (stop-mid-batch! point pool stop-by)
    (define owner (make-custodian))
    (define parked-on (zeros 256 256))
    (define behind (zeros 256 256))
    (define ts (list parked-on behind))
    (define g (handle-gate point (tensor-handle parked-on)))
    (define parked (make-watch))
    (with-parked [a (parameterize ([current-custodian owner])
                      (spawn (lambda () (tensors-free! ts)) #:pool pool #:label 'a))
                  #:at g #:watch parked]
      (check-eq? (wait-for-arrival g) 'parked)
      (case stop-by
        [(kill) (stop! a)]
        [(custodian) (custodian-shutdown-all owner)])
      (check-true (andmap tensor? ts) "a killed batched free flipped a tag")
      (check-exn #rx"being freed or was freed" (lambda () (tensor-free! parked-on)))
      (check-equal? (watch-freed-count parked (tensor-handle behind)) 0
                    "a native release ran while the killed batched free was parked")))

  (test-case "a batched free killed mid-way leaves every handle to the finalizer"
    (for* ([point (in-list '(free-taken free-unaccounted))]
           [pool (in-list pools)]
           [stop-by (in-list '(kill custodian))])
      (settle!)
      (define base (ledger-snapshot))
      (define w (make-watch))
      (with-race-hook (hooks (watch-hook w) refuse-released-finalizer)
        (stop-mid-batch! point pool stop-by)
        (settle!))
      (check-equal? (watch-finalized w) 2
                    (format "finalizers of a batched free killed at ~a (~a, ~a)"
                            point (or pool 'ordinary) stop-by))
      (check-ledger base w)))

  (define (single-against-batched parked-free other-free point pool)
    (settle!)
    (define base (ledger-snapshot))
    (define w (make-watch))
    (with-race-hook (watch-hook w)
      (define t (zeros 256 256))
      (define h (tensor-handle t))
      (define g (handle-gate point h))
      (define other-outcome
        (with-parked [a (spawn (lambda () (parked-free t)) #:pool pool #:label 'a)
                      #:at g #:watch w]
          (check-eq? (wait-for-arrival g) 'parked)
          (define outcome
            (with-handlers ([exn:fail:contract? (lambda (_) 'raised)])
              (join (spawn (lambda () (other-free t)) #:pool pool #:label 'b))
              'returned))
          (release-gate! g)
          (join a)
          outcome))
      (settle!)
      (check-equal? (watch-release-count w h) 1
                    (format "releases of a handle freed by two paths, one parked at ~a (~a)"
                            point (or pool 'ordinary)))
      (check-equal? (watch-freed-count w h) 1)
      (check-false (tensor? t))
      (check-ledger base w)
      other-outcome))

  (define (batch-free! t) (tensors-free! (list t)))

  (test-case "a batched free and an explicit free of one tensor release it once"
    (for* ([point (in-list '(free-taken free-unaccounted))]
           [pool (in-list pools)])
      (check-equal? (single-against-batched tensor-free! batch-free! point pool) 'returned
                    "a batched free racing an explicit one skips the tensor")
      (check-equal? (single-against-batched batch-free! tensor-free! point pool) 'raised
                    "an explicit free racing a batched one loses as it loses to an explicit one"))))
