#lang racket/base

(require (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in racket/contract/base -> ->* any)
         (only-in racket/match == match)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "creation-ops.rkt" zeros)
         (only-in "device-type.rkt" cpu-device cuda-device device? mps-device)
         (only-in "error.rkt" exn:fail:rktorch:oom?)
         (only-in "ops.rkt" cuda-available? mps-available? to-device))

;; The state is 'unprobed, a claim while a thread probes, then the answer.
;; The claim names its thread because kill-thread runs no dynamic-wind exit:
;; a claimant that has died holds nothing.
(struct mps-probe (run [state #:mutable]))
(struct claim (holder done))

(define (allocate-on-mps)
  (to-device (zeros 1 #:device (mps-device)) (cpu-device)))

(define/contract-out (make-mps-probe [run allocate-on-mps])
  (->* [] [(-> any)] mps-probe?)
  (mps-probe run 'unprobed))

(define ((cpu-stands-in level why) raised)
  (define detail (if (exn? raised) (exn-message raised) "a non-exn value was raised"))
  (log-message (current-logger) level 'rktorch-device
               (format "~a, so the CPU stands in: ~a" why detail)
               raised)
  #f)

(define (probe-outcome run)
  (with-handlers ([exn:fail:rktorch:oom?
                   (cpu-stands-in 'warning "MPS is available but cannot allocate")]
                  [(λ (_) #t) (cpu-stands-in 'error "the MPS probe failed")])
    (run)
    #t))

(define (claim-or-read probe)
  (call-as-atomic
   (λ ()
     (match (mps-probe-state probe)
       [(and (claim holder _) held) #:when (not (thread-dead? holder)) held]
       [(? boolean? answer) answer]
       [_ (define mine (claim (current-thread) (make-semaphore 0)))
          (set-mps-probe-state! probe mine)
          mine]))))

(define (settle! probe done)
  (define answer (probe-outcome (mps-probe-run probe)))
  (set-mps-probe-state! probe answer)
  (semaphore-post done)
  answer)

(define/contract-out (mps-usable? probe)
  (-> mps-probe? boolean?)
  (parameterize-break #f
    (match (claim-or-read probe)
      [(? boolean? answer) answer]
      [(claim (== (current-thread)) done) (settle! probe done)]
      [(claim holder done)
       (sync/enable-break (semaphore-peek-evt done) (thread-dead-evt holder))
       (mps-usable? probe)])))

(define/contract-out (pick-accelerator #:cuda? cuda? #:mps? mps? #:probe probe)
  (-> #:cuda? boolean? #:mps? boolean? #:probe mps-probe? device?)
  (cond
    [cuda? (cuda-device)]
    [(and mps? (mps-usable? probe)) (mps-device)]
    [else (cpu-device)]))

(define library-probe (make-mps-probe))

(define/contract-out (mps-if-available) (-> device?) ;; noqa
  (pick-accelerator #:cuda? #f #:mps? (mps-available?) #:probe library-probe))

(define/contract-out (accelerator-if-available) (-> device?) ;; noqa
  (pick-accelerator #:cuda? (cuda-available?)
                    #:mps? (mps-available?)
                    #:probe library-probe))
