#lang racket/base

;; Probe-local L2 candidates over the plain shim bindings, each with its own ledger.

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic
                  end-atomic
                  make-uninterruptible-lock
                  start-atomic
                  uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         "shim.rkt")

(provide (struct-out prototype)
         bare
         stage-1
         stage-1/gate
         stage-2
         alloc/uninterruptible)

;; wrap: raw op -> op over handles; handle: what the caller holds -> the
;; native pointer; free!: explicit release.
(struct prototype (name wrap handle free! live-bytes finalizer-runs))

(define (cas-add! b n)
  (let retry ()
    (define old (unbox b))
    (unless (box-cas! b old (+ old n))
      (retry))))

(struct record (state nbytes phantom))

;; A record goes live -> freed exactly once, whichever of the explicit free
;; and the finalizer gets there first. box-cas! may fail spuriously, so a
;; failure only counts as losing once the state is no longer live.
(define (claim! r)
  (define state (record-state r))
  (let retry ()
    (cond
      [(box-cas! state 'live 'freed) #t]
      [(eq? (unbox state) 'live) (retry)]
      [else #f])))

(define (make-record t)
  (define nbytes (or (shim-nbytes t) 0))
  (record (box 'live) nbytes (make-phantom-bytes nbytes)))

(define (bare)
  (prototype "bare: plain op, explicit free, no finalizer or ledger"
             values values shim-free (lambda () 0) (lambda () 0)))

(define gate-interval (* 128 1024 1024))

(define (stage-1-with name gate?)
  (define table (make-weak-hasheq))
  (define since-check (make-hasheqv))
  (define live (box 0))
  (define runs (box 0))
  (define (count-toward-gate! dev nbytes)
    (define since (+ (hash-ref since-check dev 0) nbytes))
    (hash-set! since-check dev (if (>= since gate-interval) 0 since)))
  (define (release! t find-record)
    (start-atomic)
    (define r (find-record))
    (define claimed? (and r (claim! r)))
    (when claimed?
      (hash-remove! table t)
      (set-box! live (- (unbox live) (record-nbytes r))))
    (end-atomic)
    (when claimed?
      (set-phantom-bytes! (record-phantom r) 0)
      (shim-free t)))
  (define ((finalizer-for r) t)
    (cas-add! runs 1)
    (release! t (lambda () r)))
  (define ((wrap op) . args)
    (define t (apply op args))
    (define r (make-record t))
    (define dev (and gate? (shim-device t)))
    (start-atomic)
    (register-finalizer t (finalizer-for r))
    (hash-set! table t r)
    (set-box! live (+ (unbox live) (record-nbytes r)))
    (when dev (count-toward-gate! dev (record-nbytes r)))
    (end-atomic)
    t)
  (define (free! t)
    (release! t (lambda () (hash-ref table t #f))))
  (prototype name wrap values free! (lambda () (unbox live)) (lambda () (unbox runs))))

(define (stage-1)
  (stage-1-with "stage 1: plain op, one atomic section; no pressure gate, no device query" #f))

(define (stage-1/gate)
  (stage-1-with "stage 1 + gate: device query, gate count inside the one atomic section" #t))

(struct held (handle record))

(define (stage-2)
  (define live (box 0))
  (define runs (box 0))
  (define (release! r t)
    (cas-add! live (- (record-nbytes r)))
    (set-phantom-bytes! (record-phantom r) 0)
    (shim-free t))
  (define (finalize h)
    (cas-add! runs 1)
    (define r (held-record h))
    (when (claim! r) (release! r (held-handle h))))
  (define ((wrap op) . args)
    (define t (apply op args))
    (define h (held t (make-record t)))
    (parameterize-break #f
      (register-finalizer h finalize)
      (cas-add! live (record-nbytes (held-record h))))
    h)
  (define (free! h)
    (define r (held-record h))
    (when (claim! r) (release! r (held-handle h))))
  (prototype "stage 2: plain op, no atomic section; no pressure gate, no device query"
             wrap held-handle free!
             (lambda () (unbox live)) (lambda () (unbox runs))))

(define (alloc/uninterruptible)
  (define live (box 0))
  (define runs (box 0))
  (define records (make-weak-hasheq))
  (define records-lock (make-uninterruptible-lock))
  (define (call-with-records thunk)
    (uninterruptible-lock-acquire records-lock)
    (begin0 (thunk)
            (uninterruptible-lock-release records-lock)))
  (define (unaccount! t)
    (define r
      (call-with-records
       (lambda ()
         (begin0 (hash-ref records t #f)
                 (hash-remove! records t)))))
    (when r
      (cas-add! live (- (record-nbytes r)))
      (set-phantom-bytes! (record-phantom r) 0)))
  (define (finalize t)
    (cas-add! runs 1)
    (unaccount! t)
    (shim-free t))
  (define free-handle ((deallocator #:merely-uninterruptible? #t) shim-free))
  (define (wrap op)
    (define wrapped ((allocator finalize #:merely-uninterruptible? #t) op))
    (lambda args
      (define t (apply wrapped args))
      (define r (make-record t))
      (call-with-records (lambda () (hash-set! records t r)))
      (cas-add! live (record-nbytes r))
      t))
  (define (free! t)
    (unaccount! t)
    (free-handle t))
  (prototype "ffi/unsafe/alloc #:merely-uninterruptible?; no pressure gate, no device query"
             wrap values free!
             (lambda () (unbox live)) (lambda () (unbox runs))))
