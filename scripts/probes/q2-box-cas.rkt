#lang racket/base

(require (only-in ffi/unsafe/atomic
                  call-as-atomic
                  call-as-uninterruptible
                  end-atomic
                  make-uninterruptible-lock
                  start-atomic
                  uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         (only-in math/statistics median)
         (only-in racket/match match-define)
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

(define (measure total step #:threads [n threads] #:pool [pool 'own] #:main [mode main-blocked])
  (define per-thread (quotient total n))
  (define b (box 0))
  (define-values (ms retries)
    (run-parallel n
                  (lambda (_i)
                    (for/fold ([retries 0]) ([_ (in-range per-thread)])
                      (+ retries (step b))))
                  #:pool pool
                  #:main mode))
  (list (unbox b) (apply + retries) ms))

(define hop-sections 400000)
(define hop-sections/busy 2000)

(define (hop-ms n pool total mode)
  (median < (for/list ([_ (in-range repeats)])
              (caddr (measure total atomic-increment! #:threads n #:pool pool #:main mode)))))

(define (hop-row label n pool)
  (define blocked-ms (hop-ms n pool hop-sections main-blocked))
  (define busy-ms (hop-ms n pool hop-sections/busy main-busy))
  (list label n
        hop-sections (fmt-ms blocked-ms) (fmt-ms (/ (* blocked-ms 1e6) hop-sections))
        hop-sections/busy (fmt-ms busy-ms) (fmt-ms (/ (* busy-ms 1e6) hop-sections/busy))
        (load-average)))

(define (strategy-row strategy)
  (define total (vector-ref strategy 1))
  (define samples
    (for/list ([_ (in-range repeats)]) (measure total (vector-ref strategy 2))))
  (define counts (map car samples))
  (define ms (median < (map caddr samples)))
  (list (vector-ref strategy 0)
        total
        (format "~a" counts)
        (if (andmap (lambda (c) (= c total)) counts) "yes" "NO")
        (format "~a" (map cadr samples))
        (fmt-ms ms)
        (fmt-ms (/ (* ms 1e6) total))
        (load-average)))

(define (cas-against-atomic-subtractor)
  (define per-thread (quotient full threads))
  (define b (box 0))
  (define subtractions 1000000)
  (define go (make-semaphore 0))
  (define subtractor-window (box #f))
  (define subtractor
    (thread (lambda ()
              (semaphore-wait go)
              (define start (current-inexact-monotonic-milliseconds))
              (for ([_ (in-range subtractions)])
                (start-atomic)
                (cas-add! b -1)
                (end-atomic))
              (set-box! subtractor-window
                        (cons start (current-inexact-monotonic-milliseconds))))))
  (define-values (ms adder-windows)
    (run-parallel threads
                  (lambda (i)
                    (when (zero? i) (semaphore-post go))
                    (define start (current-inexact-monotonic-milliseconds))
                    (for ([_ (in-range per-thread)]) (cas-add! b 1))
                    (cons start (current-inexact-monotonic-milliseconds)))))
  (thread-wait subtractor)
  (define adders-start (apply min (map car adder-windows)))
  (define adders-end (apply max (map cdr adder-windows)))
  (match-define (cons subtractor-start subtractor-end) (unbox subtractor-window))
  (define overlap (- (min adders-end subtractor-end) (max adders-start subtractor-start)))
  (define expected (- full subtractions))
  (define (since-adders t) (fmt-ms (- t adders-start)))
  (list "box-cas! adds from 8 parallel threads, 1M subtractions from a coroutine thread in atomic mode"
        full
        (format "(~a)" (unbox b))
        (cond
          [(not (= (unbox b) expected)) "NO"]
          [(positive? overlap) "yes"]
          [else "exact, but the subtractor never overlapped the adders"])
        (format "expected ~a; subtractor ran ~a to ~a ms, adders 0 to ~a ms"
                expected (since-adders subtractor-start) (since-adders subtractor-end)
                (since-adders adders-end))
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
   (list "call-as-atomic entered from" "threads"
         "sections, main blocked in thread-wait" "median wall ms, main blocked"
         "ns per section, main blocked"
         (format "sections, main ~a" (main-mode-label main-busy)) "median wall ms, main busy"
         "ns per section, main busy" "load"))
  (print-table-row (hop-row "coroutine thread (main OS thread)" 1 #f))
  (print-table-row (hop-row "one parallel thread" 1 'own))
  (print-table-row (hop-row "two parallel threads" 2 'own))
  (print-table-row (hop-row "eight parallel threads" 8 'own))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
