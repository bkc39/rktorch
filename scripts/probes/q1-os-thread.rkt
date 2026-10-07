#lang racket/base

;; #262 question 1: which OS thread runs each kind of protected region
;; entered from a parallel thread.

(require (only-in ffi/unsafe/atomic
                  call-as-atomic
                  call-as-uninterruptible
                  end-atomic
                  end-uninterruptible
                  make-uninterruptible-lock
                  start-atomic
                  start-uninterruptible
                  uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         "common.rkt")

(define runs 50)

(define lock (make-uninterruptible-lock))

(define regions
  (list
   (cons "plain call" (lambda (f) (f)))
   (cons "call-as-atomic" call-as-atomic)
   (cons "start-atomic / end-atomic"
         (lambda (f)
           (start-atomic)
           (begin0 (f) (end-atomic))))
   (cons "call-as-uninterruptible" call-as-uninterruptible)
   (cons "start-uninterruptible / end-uninterruptible"
         (lambda (f)
           (start-uninterruptible)
           (begin0 (f) (end-uninterruptible))))
   (cons "uninterruptible-lock held"
         (lambda (f)
           (uninterruptible-lock-acquire lock)
           (begin0 (f) (uninterruptible-lock-release lock))))
   (cons "parameterize-break #f"
         (lambda (f) (parameterize-break #f (f))))))

(define (observe enter)
  (define before (pthread-self))
  (define inside (enter pthread-self))
  (define after (pthread-self))
  (list before inside after))

(define (tally enter #:pool pool)
  (for/fold ([inside-main 0] [inside-own 0] [returned 0])
            ([_ (in-range runs)])
    (define-values (_ms results)
      (run-parallel 1 (lambda (_i) (observe enter)) #:pool pool))
    (define-values (before inside after) (apply values (car results)))
    (values (+ inside-main (if (= inside main-os-thread) 1 0))
            (+ inside-own (if (= inside before) 1 0))
            (+ returned (if (= before after) 1 0)))))

(define (region-rows label pool)
  (for/list ([region (in-list regions)])
    (define-values (inside-main inside-own returned)
      (tally (cdr region) #:pool pool))
    (list label (car region) runs inside-main inside-own returned)))

;; A thread in a shared pool can be resumed on another of the pool's OS
;; threads; this counts how often two plain calls with a yield between them
;; land on different OS threads.
(define (migrations pool threads calls)
  (define-values (_ms per-thread)
    (run-parallel threads
                  (lambda (_i)
                    (for/fold ([moved 0] [prev (pthread-self)]
                               #:result moved)
                              ([_ (in-range calls)])
                      (sleep 0)
                      (define now (pthread-self))
                      (values (if (= now prev) moved (add1 moved)) now)))
                  #:pool pool))
  (apply + per-thread))

(module+ main
  (print-banner "Q1: which OS thread runs a protected region")
  (print-table
   '("thread" "region" "runs" "inside on main OS thread"
     "inside on the thread's own OS thread" "back on own OS thread after")
   (append (region-rows "parallel, #:pool 'own" 'own)
           (region-rows "coroutine (reference)" #f)))
  (define calls 200)
  (print-table
   '("pool" "threads" "plain calls each, with a yield between"
     "calls that resumed on a different OS thread")
   (list (list "'own" 4 calls (migrations 'own 4 calls))
         (list "(make-parallel-thread-pool 2)" 4 calls
               (migrations (make-parallel-thread-pool 2) 4 calls))))
  (printf "load average (1 min) at end: ~a\n" (load-average)))
