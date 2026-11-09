#lang racket/base

(require (only-in ffi/unsafe/vm vm-eval))

(provide main-place?
         on-main-os-thread?)

(define os-thread-id (vm-eval 'get-thread-id))
(define main-os-thread
  (let ([id (box #f)])
    (thread-wait (thread (lambda () (set-box! id (os-thread-id)))))
    (unbox id)))

(define (on-main-os-thread?)
  (eqv? (os-thread-id) main-os-thread))

(define (main-place?)
  (eqv? main-os-thread 0))
