#lang racket/base

(require (only-in ffi/unsafe/atomic
                  call-as-atomic
                  call-as-uninterruptible
                  make-uninterruptible-lock)
         (only-in racket/match match-define)
         "common.rkt")

(define runs 50)

(define lock (make-uninterruptible-lock))

(define regions
  (list
   (list "plain call" (lambda (f) (f)))
   (list "call-as-atomic" call-as-atomic)
   (list "with-atomic (dynamic-wind + start-atomic / end-atomic)" (lambda (f) (with-atomic (f))))
   (list "call-as-uninterruptible" call-as-uninterruptible)
   (list "with-uninterruptible (dynamic-wind + start-uninterruptible / end-uninterruptible)"
         (lambda (f) (with-uninterruptible (f))))
   (list "with-uninterruptible-lock (dynamic-wind + lock acquire / release)"
         (lambda (f) (with-uninterruptible-lock lock (f))))
   (list "parameterize-break #f" (lambda (f) (parameterize-break #f (f))))))

(define (observe enter)
  (define before (pthread-self))
  (define inside (enter pthread-self))
  (define after (pthread-self))
  (list before inside after))

(define (tally enter #:pool pool)
  (for/fold ([inside-main 0] [inside-own 0] [returned 0])
            ([_ (in-range runs)])
    (match-define (list (list before inside after))
      (span-value (run-parallel 1 (lambda (_i) (observe enter)) #:pool pool)))
    (values (+ inside-main (if (= inside main-os-thread) 1 0))
            (+ inside-own (if (= inside before) 1 0))
            (+ returned (if (= before after) 1 0)))))

(define (region-rows label pool)
  (for/list ([region (in-list regions)])
    (match-define (list name enter) region)
    (define-values (inside-main inside-own returned) (tally enter #:pool pool))
    (list label name runs inside-main inside-own returned)))

(define (migrations pool threads calls)
  (define moves
    (run-parallel threads
                  (lambda (_i)
                    (for/fold ([moved 0] [prev (pthread-self)]
                               #:result moved)
                              ([_ (in-range calls)])
                      (sleep 0)
                      (define now (pthread-self))
                      (values (if (= now prev) moved (add1 moved)) now)))
                  #:pool pool))
  (apply + (span-value moves)))

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
  (print-footer))
