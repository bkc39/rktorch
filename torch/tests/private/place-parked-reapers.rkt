#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../../foreign/raw/cas.rkt" cas-update!)
         (only-in "../../foreign/raw/race-points.rkt" race-hook))

(provide main)

(define-runtime-path reaper "../../runtime/reaper.rkt")

(define (main ch)
  (define parked (box '()))
  (set-box! race-hook (lambda (name subject)
                        (when (eq? name 'reaper-parking)
                          (cas-update! parked (lambda (ts) (if (memq subject ts) ts (cons subject ts)))))))
  (define count (dynamic-require reaper 'reaper-count))
  (let wait ()
    (when (< (length (unbox parked)) count)
      (sleep 0.01)
      (wait)))
  (set-box! race-hook #f)
  (sleep 0.1)
  (place-channel-put ch count))
