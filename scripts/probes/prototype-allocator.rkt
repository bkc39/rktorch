#lang racket/base

;; Probe-local allocators over the plain shim bindings, the shapes the #168
;; design proposes for L2. Nothing here is library code: each keeps its own
;; ledger so a probe can check it returns to zero.
;;
;;   stage 1  the op runs plainly on the caller's OS thread, then one short
;;            atomic section registers the finalizer, the table entry and the
;;            counter
;;   stage 2  no atomic section: the finalizer is registered with breaks
;;            disabled, the counter moves by box-cas!, the record rides on
;;            the value the caller holds
;;   alloc/u  ffi/unsafe/alloc's own wrap with #:merely-uninterruptible?,
;;            and stage 2's counter
;;   bare     the op and an explicit free with no bookkeeping at all: the
;;            ceiling, and what libtorch alone does with N threads

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/alloc allocator deallocator)
         (only-in ffi/unsafe/atomic end-atomic start-atomic)
         "shim.rkt")

(provide (struct-out prototype)
         bare
         stage-1
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

(define (stage-1)
  (define table (make-weak-hasheq))
  (define live (box 0))
  (define runs (box 0))
  (define (release! r t)
    (start-atomic)
    (hash-remove! table t)
    (set-box! live (- (unbox live) (record-nbytes r)))
    (end-atomic)
    (set-phantom-bytes! (record-phantom r) 0)
    (shim-free t))
  (define ((finalizer-for r) t)
    (cas-add! runs 1)
    (when (claim! r)
      (release! r t)))
  (define ((wrap op) . args)
    (define t (apply op args))
    (define r (make-record t))
    (start-atomic)
    (register-finalizer t (finalizer-for r))
    (hash-set! table t r)
    (set-box! live (+ (unbox live) (record-nbytes r)))
    (end-atomic)
    t)
  (define (free! t)
    (define r (hash-ref table t #f))
    (when (and r (claim! r)) (release! r t)))
  (prototype "stage 1: plain op, one atomic section" wrap values free!
             (lambda () (unbox live)) (lambda () (unbox runs))))

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
  (prototype "stage 2: plain op, no atomic section" wrap held-handle free!
             (lambda () (unbox live)) (lambda () (unbox runs))))

(define (alloc/uninterruptible)
  (define live (box 0))
  (define runs (box 0))
  (define records (make-weak-hasheq))
  (define (unaccount! t)
    (define r (hash-ref records t #f))
    (when r
      (hash-remove! records t)
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
      (hash-set! records t r)
      (cas-add! live (record-nbytes r))
      t))
  (define (free! t)
    (unaccount! t)
    (free-handle t))
  (prototype "ffi/unsafe/alloc #:merely-uninterruptible?" wrap values free!
             (lambda () (unbox live)) (lambda () (unbox runs))))
