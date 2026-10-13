#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/format ~r)
         (only-in racket/match match-define)
         ;; whole-module require: only-in strips the phase-1 syntax classes
         syntax/parse/define
         "common.rkt"
         "shim.rkt")

(define repeats 3)
(define matmul-n 3072)
(define stall-window-ms 300)

(define (in-call-while call measure)
  (define inside (make-semaphore 0))
  (define-values (measured calls)
    (call-with-workers 1
                       (lambda (_i)
                         (with-final-post inside (at-set-num-threads 1))
                         (measure-span call))
                       (lambda (_workers)
                         (semaphore-wait inside)
                         (sleep 0.1)
                         (measure-span measure))))
  (match-define (list in-call) calls)
  (define in-flight (- (min (span-end in-call) (span-end measured))
                       (max (span-start in-call) (span-start measured))))
  (cons (span-value measured) (max 0.0 (/ in-flight (max (span-ms measured) 1e-3)))))

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

(define (make-calls plain-matmul)
  (list (cons "none (worker parked on a semaphore)" #f)
        (cons "usleep 1.5 s, plain _fun" (sleep-call usleep))
        (cons "usleep 1.5 s, #:blocking? #t" (sleep-call usleep/blocking))
        (cons (format "~a matmul, plain _fun" matmul-n) plain-matmul)
        (cons (format "~a matmul, #:blocking? #t" matmul-n)
              (matmul-call shim-matmul/blocking))))

(define (call-with-parked-worker thunk)
  (define release (make-semaphore 0))
  (define inside (make-semaphore 0))
  (define-values (result _)
    (call-with-workers 1
                       (lambda (_i)
                         (semaphore-post inside)
                         (semaphore-wait release))
                       (lambda (_workers)
                         (semaphore-wait inside)
                         (thunk))
                       #:release (lambda () (semaphore-post release))))
  result)

(define-syntax-parse-rule (with-parked-worker body:expr ...+)
  (call-with-parked-worker (lambda () body ...)))

(define (measure-under call measure)
  (define run
    (cond
      [call (lambda () (in-call-while call measure))]
      [else (lambda () (with-parked-worker (cons (measure) 0.0)))]))
  (define samples (repeat-runs repeats run))
  (define least-share (apply min (map cdr samples)))
  (cond
    [(and call (zero? least-share)) "call returned before the measurement"]
    [else (format "~a (~a%)"
                  (ms->string (median-of (map car samples)))
                  (~r (* 100 least-share) #:precision 1))]))

(module+ main
  (print-banner (format "Q6: collection pauses while one parallel worker is in a foreign call (median of ~a)"
                        repeats)
                #:torch-version (shim-version))
  (at-set-num-threads 1)
  (define plain-matmul (matmul-call shim-matmul))
  (define call-ms (timed plain-matmul))
  (define calls (make-calls plain-matmul))
  (printf "- one ~a matmul alone, 1 intra-op thread, on the main thread: ~a ms\n" matmul-n
          (ms->string call-ms))
  (displayln
   "- each cell: median ms (the least share of a measurement any run's call was in flight for)\n")
  (print-table-header
   (list "worker is inside" "(collect-garbage 'major) ms" "(collect-garbage 'minor) ms"
         (format "trainer's longest stall while allocating for ~a ms, ms" stall-window-ms)
         "load"))
  (for ([c (in-list calls)])
    (match-define (cons label call) c)
    (print-table-row
     (list label
           (measure-under call (lambda () (timed (lambda () (collect-garbage 'major)))))
           (measure-under call (lambda () (timed (lambda () (collect-garbage 'minor)))))
           (measure-under call longest-stall)
           (load-average))))
  (print-footer))
