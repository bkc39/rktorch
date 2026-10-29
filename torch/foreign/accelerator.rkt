#lang racket/base

(require (only-in racket/contract/base -> ->* any)
         (only-in racket/match == match)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "creation-ops.rkt" zeros)
         (only-in "device-type.rkt" cpu-device cuda-device device? mps-device)
         (only-in "error.rkt" exn:fail:rktorch:oom?)
         (only-in "ops.rkt" cuda-available? mps-available? to-device)
         (only-in "raw/fault.rkt" native-fault?)
         (only-in "raw/race-points.rkt" race-point))

(struct mps-probe (run state))
(struct claim (holder done))
(struct faulted (fault))

(define (allocate-on-mps)
  (to-device (zeros 1 #:device (mps-device)) (cpu-device)))

(define/contract-out (make-mps-probe [run allocate-on-mps])
  (->* [] [(-> any)] mps-probe?)
  (mps-probe run (box 'unprobed)))

(define ((cpu-stands-in level why) raised)
  (define detail (if (exn? raised) (exn-message raised) "a non-exn value was raised"))
  (log-message (current-logger) level 'rktorch-device
               (format "~a, so the CPU stands in: ~a" why detail)
               raised)
  #f)

(define (probe-outcome run)
  (with-handlers ([native-fault? faulted]
                  [exn:fail:rktorch:oom?
                   (cpu-stands-in 'warning "MPS is available but cannot allocate")]
                  [(λ (_) #t) (cpu-stands-in 'error "the MPS probe failed")])
    (run)
    #t))

;; A CAS, not an atomic section, so a parallel thread decides it on its own
;; OS thread; losing it means another thread changed the state, so read again.
(define (claim-or-read probe)
  (define state (mps-probe-state probe))
  (define seen (unbox state))
  (match seen
    [(claim holder _) #:when (not (thread-dead? holder)) seen]
    [(? boolean? answer) answer]
    [(? faulted? fault) fault]
    [_ (define mine (claim (current-thread) (make-semaphore 0)))
       (if (box-cas! state seen mine)
           mine
           (claim-or-read probe))]))

(define (settle! probe done)
  (race-point mps-probe-claimed probe)
  (set-box! (mps-probe-state probe) (probe-outcome (mps-probe-run probe)))
  (race-point mps-probe-settled probe)
  (semaphore-post done)
  (mps-usable? probe))

(define/contract-out (mps-usable? probe)
  (-> mps-probe? boolean?)
  (parameterize-break #f
    (match (claim-or-read probe)
      [(? boolean? answer) answer]
      [(faulted fault) (raise fault)]
      [(claim (== (current-thread)) done) (settle! probe done)]
      [(claim holder done)
       (race-point mps-probe-waiting probe)
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
