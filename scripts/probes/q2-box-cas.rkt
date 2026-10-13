#lang racket/base

(require (only-in ffi/unsafe/atomic
                  call-as-atomic
                  call-as-uninterruptible
                  make-uninterruptible-lock)
         (only-in racket/format ~r)
         (only-in racket/list append*)
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
   (strategy "with-uninterruptible-lock (dynamic-wind + lock acquire / release)" full
             (lambda (b) (with-uninterruptible-lock lock (bump! b))))
   (strategy "call-as-uninterruptible alone" full
             (lambda (b) (call-as-uninterruptible (lambda () (bump! b)))))
   (strategy "call-as-atomic" (quotient full 10) atomic-increment!)
   (strategy "with-atomic (dynamic-wind + start-atomic / end-atomic)" (quotient full 10)
             (lambda (b) (with-atomic (bump! b))))))

(struct sample (count retries ms woke))

(define (measure total step #:threads [n threads] #:pool [pool 'own] #:main [mode main-blocked])
  (define per-thread (quotient total n))
  (define b (box 0))
  (define woke (box #f))
  (define window
    (run-parallel n
                  (lambda (_i)
                    (for/fold ([retries 0]) ([_ (in-range per-thread)])
                      (+ retries (step b))))
                  #:pool pool
                  #:main mode
                  #:on-wake (lambda () (set-box! woke (unbox b)))))
  (sample (unbox b) (apply + (span-value window)) (span-ms window) (unbox woke)))

(define (ns-per ms count)
  (~> ms (* 1e6) (/ count) (~r #:precision 1)))

(define hop-sections 400000)
(define hop-sections/busy 2000)
(define hop-sections/foreign 200)
(define hop-foreign-seconds 0.5)

(define hop-modes
  (list (cons main-blocked hop-sections)
        (cons main-busy hop-sections/busy)
        (cons (main-in-foreign hop-foreign-seconds) hop-sections/foreign)))

(define (hop-columns n pool total mode)
  (define samples
    (repeat-runs repeats
                 (lambda () (measure total atomic-increment! #:threads n #:pool pool #:main mode))))
  (define ms (median-of (map sample-ms samples)))
  (define woke (map sample-woke samples))
  (list total
        (ms->string ms)
        (ns-per ms total)
        (if (andmap values woke) (median-of woke) "-")))

(define (hop-row label n pool)
  (append (list label n)
          (append*
           (for/list ([hop (in-list hop-modes)])
             (match-define (cons mode sections) hop)
             (hop-columns n pool sections mode)))
          (list (load-average))))

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
        (ms->string ms)
        (ns-per ms total)
        (load-average)))

(define (cas-against-atomic-subtractor)
  (define per-thread (quotient full threads))
  (define b (box 0))
  (define subtractions 1000000)
  (define go (make-semaphore 0))
  (define-values (adders subtractor)
    (call-with-workers 1
                       (lambda (_i)
                         (semaphore-wait go)
                         (measure-span (lambda ()
                                         (for ([_ (in-range subtractions)])
                                           (with-atomic (cas-update! b sub1))))))
                       (lambda (_subtractor)
                         (run-parallel threads
                                       (lambda (i)
                                         (when (zero? i) (semaphore-post go))
                                         (for ([_ (in-range per-thread)]) (cas-update! b add1))
                                         (now-ms))))
                       #:pool #f
                       #:release (lambda () (semaphore-post go))))
  (match-define (list subtracted) subtractor)
  (define adders-start (span-start adders))
  (define adders-end (apply max (span-value adders)))
  (define overlap (- (min adders-end (span-end subtracted))
                     (max adders-start (span-start subtracted))))
  (define expected (- full subtractions))
  (define (since-adders t) (ms->string (- t adders-start)))
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
        (ms->string (span-ms adders))
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
  (displayln
   (string-append
    "- with main in foreign calls, wall time includes the foreign phase; the count at main's last"
    " wake is what the threads finished while main was in foreign calls\n"))
  (print-table-header
   (append '("call-as-atomic entered from" "threads")
           (append*
            (for/list ([hop (in-list hop-modes)])
              (define main (main-mode-label (car hop)))
              (list (format "sections, main ~a" main)
                    (format "median wall ms, main ~a" main)
                    (format "ns per section, main ~a" main)
                    (format "sections done at main's last wake, main ~a" main))))
           '("load")))
  (print-table-row
   (hop-row "coroutine thread (main OS thread; while main is busy or in foreign calls this times sharing main's time slices, not the section)"
            1 #f))
  (print-table-row (hop-row "one parallel thread" 1 'own))
  (print-table-row (hop-row "two parallel threads" 2 'own))
  (print-table-row (hop-row "eight parallel threads" 8 'own))
  (print-footer))
