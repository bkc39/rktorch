#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in "../../foreign/raw/collector.rkt" collect-and-wait!)
         (only-in "../../main.rkt" ones))

(provide run-race)

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
    ;; a spin, not a semaphore: woken one at a time, the workers reach their
    ;; first registration too far apart to race it
    (let spin ()
      (unless (unbox go)
        (spin)))
    (collect-and-wait!)
    (void (ones 2 2))
    (for ([_ (in-range per-worker)])
      (register-finalizer (box 0) record!)))
  (define threads
    (for/list ([_ (in-range workers)])
      (thread work #:pool 'own)))
  (for ([_ (in-range workers)])
    (sync/timeout (remaining) ready))
  (set-box! go #t)
  (for ([t (in-list threads)])
    (sync/timeout (remaining) t))
  (define total (* workers per-worker))
  (define finalized
    (let drain ([n 0])
      (cond
        [(= n total) n]
        [(sync/timeout 0.05 ran) (drain (add1 n))]
        [(positive? (remaining)) (collect-garbage) (drain n)]
        [else n])))
  (list finalized (hash-count runners)))
