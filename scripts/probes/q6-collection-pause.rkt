#lang racket/base

;; #262 question 6: how long a collection waits while one parallel worker is
;; inside a long foreign call, plain against #:blocking? #t.

(require "common.rkt"
         "shim.rkt")

(define repeats 3)
(define matmul-n 3072)
(define stall-window-ms 300)

;; Answers the measurement and the share of it the call was in flight for,
;; from the worker's last timestamp before the call to its first after. A
;; plain call holds the collection until it returns, so its share reads 1
;; even though it returns just before the measurement ends.
(define (in-call-while call measure)
  (define inside (make-semaphore 0))
  (define entered-at (box #f))
  (define returned-at (box #f))
  (define failure (box #f))
  (define worker
    (thread (lambda ()
              (at-set-num-threads 1)
              (semaphore-post inside)
              (with-handlers ([(lambda (_) #t) (lambda (e) (set-box! failure e))])
                (set-box! entered-at (current-inexact-monotonic-milliseconds))
                (call))
              (set-box! returned-at (current-inexact-monotonic-milliseconds)))
            #:pool 'own))
  (semaphore-wait inside)
  (sleep 0.1)
  (define started-at (current-inexact-monotonic-milliseconds))
  (define result (measure))
  (define ended-at (current-inexact-monotonic-milliseconds))
  (thread-wait worker)
  (when (unbox failure)
    (raise (unbox failure)))
  (define in-flight (- (min (unbox returned-at) ended-at)
                       (max (unbox entered-at) started-at)))
  (cons result (max 0.0 (/ in-flight (max (- ended-at started-at) 1e-3)))))

(define (timed thunk)
  (define start (current-inexact-monotonic-milliseconds))
  (thunk)
  (- (current-inexact-monotonic-milliseconds) start))

;; The trainer's view: allocate for a while and report the longest gap
;; between two iterations, which is how long it was held at a collection.
(define sink (box #f))

(define (longest-stall)
  (define end (+ (current-inexact-monotonic-milliseconds) stall-window-ms))
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

;; Built after the intra-op count is set, since the matmul inputs are the
;; first libtorch ops.
(define (make-calls)
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
  (begin0 (cons (measure) 0.0)
          (semaphore-post release)
          (thread-wait worker)))

(define (measure-under call measure)
  (define samples
    (for/list ([_ (in-range repeats)])
      (cond
        [call (in-call-while call measure)]
        [else (parked measure)])))
  (define least-share (apply min (map cdr samples)))
  (cond
    [(and call (zero? least-share)) "call returned before the measurement"]
    [else (format "~a (~a%)"
                  (fmt-ms (median (map car samples)))
                  (fmt-ms (* 100 least-share)))]))

(module+ main
  (print-banner (format "Q6: collection pauses while one parallel worker is in a foreign call (median of ~a)"
                        repeats)
                #:torch-version (shim-version))
  (at-set-num-threads 1)
  (define calls (make-calls))
  (define call-ms (timed (cdr (list-ref calls 3))))
  (printf "- one ~a matmul alone, 1 intra-op thread, on the main thread: ~a ms\n" matmul-n
          (fmt-ms call-ms))
  (displayln
   "- each cell: median ms (the least share of a measurement any run's call was in flight for)\n")
  (print-table-header
   (list "worker is inside" "(collect-garbage 'major) ms" "(collect-garbage 'minor) ms"
         (format "trainer's longest stall while allocating for ~a ms, ms" stall-window-ms)
         "load"))
  (for ([c (in-list calls)])
    (define call (cdr c))
    (print-table-row
     (list (car c)
           (measure-under call (lambda () (timed (lambda () (collect-garbage 'major)))))
           (measure-under call (lambda () (timed (lambda () (collect-garbage 'minor)))))
           (measure-under call longest-stall)
           (load-average))))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
