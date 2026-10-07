#lang racket/base

;; #262 question 2: is box-cas! exact across parallel threads, against the
;; other ways a shared counter could be guarded?

(require (only-in ffi/unsafe/atomic
                  call-as-atomic
                  call-as-uninterruptible
                  end-atomic
                  make-uninterruptible-lock
                  start-atomic
                  uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         "common.rkt")

(define threads 8)
(define full 4000000)
(define repeats 3)

(define (cas-add! b n)
  (let retry ([tries 0])
    (define old (unbox b))
    (cond
      [(box-cas! b old (+ old n)) tries]
      [else (retry (add1 tries))])))

(define lock (make-uninterruptible-lock))

(define (bump! b)
  (set-box! b (add1 (unbox b)))
  0)

(define (atomic-increment! b)
  (call-as-atomic (lambda () (bump! b))))

;; call-as-atomic hops to the main OS thread for every increment, so it gets
;; a tenth of the increments to finish in minutes rather than hours.
(define strategies
  (list
   (vector "unprotected unbox / set-box!" full bump!)
   (vector "box-cas! retry loop" full (lambda (b) (cas-add! b 1)))
   (vector "uninterruptible lock" full
           (lambda (b)
             (uninterruptible-lock-acquire lock)
             (bump! b)
             (uninterruptible-lock-release lock)
             0))
   (vector "call-as-uninterruptible alone" full
           (lambda (b) (call-as-uninterruptible (lambda () (bump! b)))))
   (vector "call-as-atomic" (quotient full 10) atomic-increment!)
   (vector "start-atomic / end-atomic" (quotient full 10)
           (lambda (b)
             (start-atomic)
             (bump! b)
             (end-atomic)
             0))))

(define (measure total step #:threads [n threads] #:pool [pool 'own])
  (define per-thread (quotient total n))
  (define b (box 0))
  (define-values (ms retries)
    (run-parallel n
                  (lambda (_i)
                    (for/fold ([retries 0]) ([_ (in-range per-thread)])
                      (+ retries (step b))))
                  #:pool pool))
  (list (unbox b) (apply + retries) ms))

;; What one atomic section costs by where it is entered from: the main OS
;; thread's coroutine, a lone parallel thread, and 8 of them contending.
(define (hop-row label n pool total)
  (define ms (median (for/list ([_ (in-range repeats)])
                       (caddr (measure total atomic-increment!
                                       #:threads n #:pool pool)))))
  (list label n total (fmt-ms ms) (fmt-ms (/ (* ms 1e6) total)) (load-average)))

(define (strategy-row strategy)
  (define total (vector-ref strategy 1))
  (define samples
    (for/list ([_ (in-range repeats)]) (measure total (vector-ref strategy 2))))
  (define counts (map car samples))
  (define ms (median (map caddr samples)))
  (list (vector-ref strategy 0)
        total
        (format "~a" counts)
        (if (andmap (lambda (c) (= c total)) counts) "yes" "NO")
        (format "~a" (map cadr samples))
        (fmt-ms ms)
        (fmt-ms (/ (* ms 1e6) total))
        (load-average)))

;; The Stage 2 ledger's mix: parallel workers add while a coroutine thread in
;; atomic mode, as a finalizer runs, subtracts.
(define (cas-against-atomic-subtractor)
  (define per-thread (quotient full threads))
  (define b (box 0))
  (define subtractions 1000000)
  (define go (make-semaphore 0))
  (define subtractor
    (thread (lambda ()
              (semaphore-wait go)
              (for ([_ (in-range subtractions)])
                (start-atomic)
                (cas-add! b -1)
                (end-atomic)))))
  (define-values (ms _retries)
    (run-parallel threads
                  (lambda (i)
                    (when (zero? i) (semaphore-post go))
                    (for ([_ (in-range per-thread)]) (cas-add! b 1)))))
  (thread-wait subtractor)
  (define expected (- full subtractions))
  (list "box-cas! adds from 8 parallel threads, 1M subtractions from a coroutine thread in atomic mode"
        full
        (format "(~a)" (unbox b))
        (if (= (unbox b) expected) "yes" "NO")
        (format "expected ~a" expected)
        (fmt-ms ms)
        ""
        (load-average)))

(module+ main
  (print-banner (format "Q2: increments of one shared counter over ~a parallel threads, ~a runs each"
                        threads repeats))
  (print-table-header
   '("guard" "increments" "final count per run" "exact" "CAS retries per run"
     "median wall ms" "ns per increment" "load"))
  (for ([s (in-list strategies)])
    (print-table-row (strategy-row s)))
  (print-table-row (cas-against-atomic-subtractor))
  (newline)
  (print-table-header
   '("call-as-atomic entered from" "threads" "sections" "median wall ms"
     "ns per section" "load"))
  (print-table-row (hop-row "coroutine thread (main OS thread)" 1 #f 400000))
  (print-table-row (hop-row "one parallel thread" 1 'own 400000))
  (print-table-row (hop-row "two parallel threads" 2 'own 400000))
  (print-table-row (hop-row "eight parallel threads" 8 'own 400000))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
