#lang racket/base

(require (only-in ffi/unsafe/atomic
                  call-as-atomic
                  call-as-uninterruptible
                  make-uninterruptible-lock)
         (only-in racket/match match-define)
         (only-in threading ~>)
         "common.rkt")

(define threads 8)
(define full 4000000)
(define repeats 3)

(define lock (make-uninterruptible-lock))

(define (bump! b)
  (set-box! b (add1 (unbox b)))
  0)

(define (atomic-increment! b)
  (call-as-atomic (lambda () (bump! b))))

(struct strategy (label total step))

(define strategies
  (list
   (strategy "unprotected unbox / set-box!" full bump!)
   (strategy "box-cas! retry loop" full (lambda (b) (cas-update! b add1)))
   (strategy "with-uninterruptible-lock" full
             (lambda (b) (with-uninterruptible-lock lock (bump! b))))
   (strategy "call-as-uninterruptible alone" full
             (lambda (b) (call-as-uninterruptible (lambda () (bump! b)))))
   (strategy "call-as-atomic" (quotient full 10) atomic-increment!)
   (strategy "with-atomic (start-atomic / end-atomic)" (quotient full 10)
             (lambda (b) (with-atomic (bump! b))))))

(struct sample (count retries ms))

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
  (sample (unbox b) (apply + retries) ms))

(define (ns-per ms count)
  (~> ms (* 1e6) (/ count) fmt-ms))

(define hop-sections 400000)
(define hop-sections/busy 2000)

(define (hop-ms n pool total mode)
  (median-of-runs repeats
                  (lambda ()
                    (sample-ms (measure total atomic-increment! #:threads n #:pool pool #:main mode)))))

(define (hop-row label n pool)
  (define blocked-ms (hop-ms n pool hop-sections main-blocked))
  (define busy-ms (hop-ms n pool hop-sections/busy main-busy))
  (list label n
        hop-sections (fmt-ms blocked-ms) (ns-per blocked-ms hop-sections)
        hop-sections/busy (fmt-ms busy-ms) (ns-per busy-ms hop-sections/busy)
        (load-average)))

(define (strategy-row s)
  (match-define (strategy label total step) s)
  (define samples (repeat-runs repeats (lambda () (measure total step))))
  (define counts (map sample-count samples))
  (define ms (median-of (map sample-ms samples)))
  (list label
        total
        (format "~a" counts)
        (if (andmap (lambda (c) (= c total)) counts) "yes" "NO")
        (format "~a" (map sample-retries samples))
        (fmt-ms ms)
        (ns-per ms total)
        (load-average)))

(define (cas-against-atomic-subtractor)
  (define per-thread (quotient full threads))
  (define b (box 0))
  (define subtractions 1000000)
  (define go (make-semaphore 0))
  (define subtractor-span (box #f))
  (define subtractor
    (thread (lambda ()
              (semaphore-wait go)
              (set-box! subtractor-span
                        (measure-span (lambda ()
                                        (for ([_ (in-range subtractions)])
                                          (with-atomic (cas-update! b sub1)))))))))
  (define adders
    (run-parallel/gc threads
                     (lambda (i)
                       (when (zero? i) (semaphore-post go))
                       (for ([_ (in-range per-thread)]) (cas-update! b add1))
                       (now-ms))))
  (thread-wait subtractor)
  (define adders-start (span-start adders))
  (define adders-end (apply max (span-value adders)))
  (define subtracted (unbox subtractor-span))
  (define overlap (- (min adders-end (span-end subtracted))
                     (max adders-start (span-start subtracted))))
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
                expected (since-adders (span-start subtracted)) (since-adders (span-end subtracted))
                (since-adders adders-end))
        (fmt-ms (span-ms adders))
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
  (print-footer))
