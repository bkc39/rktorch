#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in "../../foreign.rkt" + cpu-device native-memory-use sum zeros)
         (only-in "../../foreign/raw/collector.rkt"
                  collect-and-wait! finalizer-runs on-main-os-thread?)
         (only-in "../../foreign/raw/race-points.rkt" race-hook)
         (only-in (submod "../../runtime/reaper.rkt" private) reaper-threads))

(provide main)

(define ops 2000)
(define op-bytes (* 64 64 4))

(define (cpu-bytes)
  (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
        [else 0]))

(define (main ch)
  (define off-main (box 0))
  (set-box! race-hook (lambda (name _subject)
                        (when (and (eq? name 'finalizer-releasing) (not (on-main-os-thread?)))
                          (set-box! off-main (add1 (unbox off-main))))))
  (zeros 1)
  (collect-and-wait!)
  (define baseline (cpu-bytes))
  (define runs-before (finalizer-runs))
  (for ([_ (in-range ops)])
    (sum (+ (zeros 64 64) 1.0)))
  (define drained? (collect-and-wait!))
  (set-box! race-hook #f)
  (place-channel-put ch (list (null? (reaper-threads))
                              drained?
                              (<= (cpu-bytes) (+ baseline (* 2 op-bytes)))
                              (>= (- (finalizer-runs) runs-before) ops)
                              (zero? (unbox off-main)))))
