#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in "../../foreign/raw/collector.rkt" collect-and-wait!)
         (only-in "../../main.rkt" ones)
         (only-in "race-harness.rkt" join-all spawn))

(provide run-race)

(define (spin-until-released go)
  (unless (unbox go)
    (spin-until-released go)))

(define (run-race workers per-worker)
  (define deadline (+ (current-inexact-milliseconds) 30000))
  (define (remaining)
    (max 0 (/ (- deadline (current-inexact-milliseconds)) 1000.0)))
  (define runners (make-hasheq))
  (define ran (make-semaphore 0))
  (define (record! _)
    (hash-set! runners (current-thread) #t)
    (semaphore-post ran))
  (define ready (make-semaphore 0))
  (define go (box #f))
  (define (work)
    (semaphore-post ready)
    (spin-until-released go)
    (collect-and-wait!)
    (void (ones 2 2))
    (for ([_ (in-range per-worker)])
      (register-finalizer (box 0) record!)))
  (define spawned
    (for/list ([_ (in-range workers)])
      (spawn work #:pool 'own)))
  (for ([_ (in-range workers)])
    (sync/timeout (remaining) ready))
  (set-box! go #t)
  (join-all spawned #:seconds (remaining))
  (define total (* workers per-worker))
  (define finalized
    (let drain ([n 0])
      (cond
        [(= n total) n]
        [(sync/timeout 0.05 ran) (drain (add1 n))]
        [(positive? (remaining)) (collect-garbage) (drain n)]
        [else n])))
  (list finalized (hash-count runners)))
