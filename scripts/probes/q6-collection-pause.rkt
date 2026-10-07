#lang racket/base

;; #262 question 6: how long a collection waits while one parallel worker is
;; inside a long foreign call, plain against #:blocking? #t.

(require "common.rkt"
         "shim.rkt")

(define repeats 3)
(define matmul-n 3072)

(define (in-call-while call measure)
  (define inside (make-semaphore 0))
  (define worker
    (thread (lambda ()
              (at-set-num-threads 1)
              (semaphore-post inside)
              (call))
            #:pool 'own))
  (semaphore-wait inside)
  (sleep 0.1)
  (begin0 (measure)
          (thread-wait worker)))

(define (timed thunk)
  (define start (current-inexact-monotonic-milliseconds))
  (thunk)
  (- (current-inexact-monotonic-milliseconds) start))

;; The trainer's view: allocate for a second and report the longest gap
;; between two iterations, which is how long it was held at a collection.
(define (longest-stall)
  (define end (+ (current-inexact-monotonic-milliseconds) 1000))
  (let loop ([last (current-inexact-monotonic-milliseconds)] [worst 0.0] [sink #f])
    (define now (current-inexact-monotonic-milliseconds))
    (define worst* (max worst (- now last)))
    (cond
      [(> now end) worst*]
      [else (loop now worst* (make-vector 256 sink))])))

(define (sleep-call usleep-fn) (lambda () (usleep-fn 1500000)))

(define (matmul-call matmul-fn)
  (define a (shim-randn matmul-n matmul-n))
  (lambda ()
    (shim-free (matmul-fn a a))))

(define calls
  (list (cons "none (worker parked on a semaphore)" #f)
        (cons "usleep 1.5 s, plain _fun" (sleep-call usleep))
        (cons "usleep 1.5 s, #:blocking? #t" (sleep-call usleep/blocking))
        (cons (format "~a matmul, plain _fun" matmul-n) (matmul-call shim-matmul))
        (cons (format "~a matmul, #:blocking? #t" matmul-n)
              (matmul-call shim-matmul/blocking))))

(define (parked measure)
  (define release (make-semaphore 0))
  (define inside (make-semaphore 0))
  (define worker (thread (lambda () (semaphore-post inside) (semaphore-wait release))
                         #:pool 'own))
  (semaphore-wait inside)
  (begin0 (measure)
          (semaphore-post release)
          (thread-wait worker)))

(define (measure-under call measure)
  (median
   (for/list ([_ (in-range repeats)])
     (cond
       [call (in-call-while call measure)]
       [else (parked measure)]))))

(module+ main
  (print-banner (format "Q6: collection pauses while one parallel worker is in a foreign call (median of ~a)"
                        repeats)
                #:torch-version (shim-version))
  (at-set-num-threads 1)
  (define call-ms (timed (cdr (list-ref calls 3))))
  (printf "- one ~a matmul alone, 1 intra-op thread, on the main thread: ~a ms\n\n"
          matmul-n (fmt-ms call-ms))
  (print-table-header
   '("worker is inside" "(collect-garbage 'major) ms" "(collect-garbage 'minor) ms"
     "trainer's longest stall while allocating for 1 s, ms" "load"))
  (for ([c (in-list calls)])
    (define call (cdr c))
    (print-table-row
     (list (car c)
           (fmt-ms (measure-under call (lambda () (timed (lambda () (collect-garbage 'major))))))
           (fmt-ms (measure-under call (lambda () (timed (lambda () (collect-garbage 'minor))))))
           (fmt-ms (measure-under call longest-stall))
           (load-average))))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
