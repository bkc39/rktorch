#lang racket/base

(require (only-in ffi/unsafe/vm vm-eval))

(provide main-place?
         on-main-os-thread?)

;; Green threads run on the place's main OS thread, whichever thread
;; instantiates this module.
(define os-thread-id (vm-eval 'get-thread-id))
(define main-os-thread
  (let ([id (box #f)])
    (thread-wait (thread (lambda () (set-box! id (os-thread-id)))))
    (unbox id)))

(define (on-main-os-thread?)
  (eqv? (os-thread-id) main-os-thread))

;; Chez numbers the process's first OS thread 0, and only the main place
;; runs its green threads there.
(define (main-place?)
  (eqv? main-os-thread 0))
