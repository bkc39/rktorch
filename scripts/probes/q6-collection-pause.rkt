#lang racket/base

;; #262 question 6: how long a collection waits while one parallel worker is
;; inside a long foreign call, plain against #:blocking? #t.

(require "common.rkt"
         "shim.rkt")

(define repeats 3)
(define matmul-n 3072)

;; #f when the call returned before the measurement started, so nothing
;; overlapped it.
(define (in-call-while call measure)
  (define inside (make-semaphore 0))
  (define returned-at (box #f))
  (define worker
    (thread (lambda ()
              (at-set-num-threads 1)
              (semaphore-post inside)
              (call)
              (set-box! returned-at (current-inexact-monotonic-milliseconds)))
            #:pool 'own))
  (semaphore-wait inside)
  (sleep 0.1)
  (define started-at (current-inexact-monotonic-milliseconds))
  (define result (measure))
  (thread-wait worker)
  (and (> (unbox returned-at) started-at) result))

(define (timed thunk)
  (define start (current-inexact-monotonic-milliseconds))
  (thunk)
  (- (current-inexact-monotonic-milliseconds) start))

;; The trainer's view: allocate for a second and report the longest gap
;; between two iterations, which is how long it was held at a collection.
(define sink (box #f))

(define (longest-stall)
  (define end (+ (current-inexact-monotonic-milliseconds) 1000))
  (let loop ([last (current-inexact-monotonic-milliseconds)] [worst 0.0])
    (define now (current-inexact-monotonic-milliseconds))
    (define worst* (max worst (- now last)))
    (set-box! sink (make-vector 256 #f))
    (cond
      [(> now end) worst*]
      [else (loop now worst*)])))

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
  (define samples
    (for/list ([_ (in-range repeats)])
      (cond
        [call (in-call-while call measure)]
        [else (parked measure)])))
  (cond
    [(andmap values samples) (fmt-ms (median samples))]
    [else "call returned before the measurement"]))

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
           (measure-under call (lambda () (timed (lambda () (collect-garbage 'major)))))
           (measure-under call (lambda () (timed (lambda () (collect-garbage 'minor)))))
           (measure-under call longest-stall)
           (load-average))))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
