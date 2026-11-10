#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in racket/place place-channel-put)
         (only-in racket/runtime-path #%datum define-runtime-path))

(provide main)

(define-runtime-path collector "../../foreign/raw/collector.rkt")

(define (finalizer-runs-on-main? on-main?)
  (define ran-on-main (box 'unrun))
  (thread-wait
   (thread (lambda ()
             (register-finalizer (box 0) (lambda (_) (set-box! ran-on-main (on-main?)))))
           #:pool 'own))
  (for ([_ (in-range 100)] #:break (boolean? (unbox ran-on-main)))
    (collect-garbage)
    (sync/timeout 0.02 never-evt))
  (unbox ran-on-main))

(define (main ch)
  (define on-main? (box #f))
  (define off-main (box 'unread))
  (thread-wait
   (thread (lambda ()
             (define p (dynamic-require collector 'on-main-os-thread?))
             (set-box! on-main? p)
             (set-box! off-main (p)))
           #:pool 'own))
  (define p (unbox on-main?))
  (place-channel-put ch (list (p) (unbox off-main) (finalizer-runs-on-main? p))))
