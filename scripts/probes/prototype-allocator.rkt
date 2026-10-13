#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic make-uninterruptible-lock)
         (only-in racket/function const)
         (only-in "common.rkt"
                  cas-update!
                  with-atomic
                  with-finalizer-in-flight
                  with-uninterruptible-lock)
         "shim.rkt")

(provide (struct-out prototype)
         bare
         stage-1
         stage-1/gate
         stage-2
         alloc/uninterruptible)

(struct prototype (name wrap handle free! live-bytes finalizer-frees finalizer-calls))

(struct record (state nbytes phantom))

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
             values values shim-free (const 0) #f #f))

(define gate-interval (* 128 1024 1024))

(define (stage-1-with name gate?)
  (define table (make-weak-hasheq))
  (define since-check (make-hasheqv))
  (define live (box 0))
  (define frees (box 0))
  (define calls (box 0))
  (define (count-toward-gate! dev nbytes)
    (define since (+ (hash-ref since-check dev 0) nbytes))
    (hash-set! since-check dev (if (>= since gate-interval) 0 since)))
  (define (claim-record! t find-record)
    (with-atomic
      (define r (find-record))
      (cond
        [(and r (claim! r))
         (hash-remove! table t)
         (set-box! live (- (unbox live) (record-nbytes r)))
         r]
        [else #f])))
  (define (release! r t)
    (set-phantom-bytes! (record-phantom r) 0)
    (shim-free t))
  (define ((finalizer-for r) t)
    (with-finalizer-in-flight
      (cas-update! calls add1)
      (when (claim-record! t (const r))
        (cas-update! frees add1)
        (release! r t))))
  (define ((wrap op) . args)
    (define t (apply op args))
    (define r (make-record t))
    (with-atomic
      (define dev (and gate? (shim-device t)))
      (register-finalizer t (finalizer-for r))
      (hash-set! table t r)
      (set-box! live (+ (unbox live) (record-nbytes r)))
      (when dev (count-toward-gate! dev (record-nbytes r))))
    t)
  (define (free! t)
    (define r (claim-record! t (lambda () (hash-ref table t #f))))
    (when r (release! r t)))
  (prototype name wrap values free!
             (lambda () (unbox live)) (lambda () (unbox frees)) (lambda () (unbox calls))))

(define (stage-1)
  (stage-1-with "stage 1: plain op, one atomic section; no pressure gate, no device query" #f))

(define (stage-1/gate)
  (stage-1-with "stage 1 + gate: device query, gate count inside the one atomic section" #t))

(struct held (handle record))

(define (stage-2)
  (define live (box 0))
  (define frees (box 0))
  (define calls (box 0))
  (define (release! r t)
    (cas-update! live - (record-nbytes r))
    (set-phantom-bytes! (record-phantom r) 0)
    (shim-free t))
  (define (finalize h)
    (with-finalizer-in-flight
      (cas-update! calls add1)
      (define r (held-record h))
      (when (claim! r)
        (cas-update! frees add1)
        (release! r (held-handle h)))))
  (define ((wrap op) . args)
    (define t (apply op args))
    (define h (held t (make-record t)))
    (parameterize-break #f
      (register-finalizer h finalize)
      (cas-update! live + (record-nbytes (held-record h))))
    h)
  (define (free! h)
    (define r (held-record h))
    (when (claim! r) (release! r (held-handle h))))
  (prototype "stage 2: plain op, no atomic section; no pressure gate, no device query"
             wrap held-handle free!
             (lambda () (unbox live)) (lambda () (unbox frees)) (lambda () (unbox calls))))

(define (alloc/uninterruptible)
  (define live (box 0))
  (define runs (box 0))
  (define records (make-weak-hasheq))
  (define records-lock (make-uninterruptible-lock))
  (define (unaccount! t)
    (define r
      (with-uninterruptible-lock records-lock
        (begin0 (hash-ref records t #f)
                (hash-remove! records t))))
    (when r
      (cas-update! live - (record-nbytes r))
      (set-phantom-bytes! (record-phantom r) 0)))
  (define (finalize t)
    (with-finalizer-in-flight
      (cas-update! runs add1)
      (unaccount! t)
      (shim-free t)))
  (define free-handle ((deallocator #:merely-uninterruptible? #t) shim-free))
  (define (wrap op)
    (define wrapped ((allocator finalize #:merely-uninterruptible? #t) op))
    (lambda args
      (define t (apply wrapped args))
      (define r (make-record t))
      (with-uninterruptible-lock records-lock (hash-set! records t r))
      (cas-update! live + (record-nbytes r))
      t))
  (define (free! t)
    (unaccount! t)
    (free-handle t))
  (prototype "ffi/unsafe/alloc #:merely-uninterruptible?; no pressure gate, no device query"
             wrap values free!
             (lambda () (unbox live)) (lambda () (unbox runs)) #f))
