#lang racket/base

(require (only-in racket/place place-channel-put)
         (only-in racket/runtime-path #%datum define-runtime-path))

(provide main)

(define-runtime-path collector "../../foreign/raw/collector.rkt")

(define (main ch)
  (define on-main? (box #f))
  (define off-main (box 'unread))
  (thread-wait
   (thread (lambda ()
             (define p (dynamic-require collector 'on-main-os-thread?))
             (set-box! on-main? p)
             (set-box! off-main (p)))
           #:pool 'own))
  (place-channel-put ch (list ((unbox on-main?)) (unbox off-main))))
