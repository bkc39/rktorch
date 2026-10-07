#lang racket/base

;; #262 question 5: tensor-making throughput from N parallel workers through
;; today's allocator against the probe-local prototypes, which run the op
;; outside atomic mode.

(require (only-in racket/format ~r)
         "common.rkt"
         "prototype-allocator.rkt"
         "shim.rkt"
         "variants.rkt")

(define (iteration! p o n)
  (define h (prototype-handle p))
  (define free! (prototype-free! p))
  (define a ((ops-randn o) n))
  (define b ((ops-add o) (h a) (h a)))
  (define c ((ops-matmul o) (h b) (h a)))
  (free! c)
  (free! b)
  (free! a))

(define tensors-per-iteration 3)

(define (loop-until deadline p o n)
  (let loop ([iterations 0])
    (cond
      [(> (current-inexact-monotonic-milliseconds) deadline) iterations]
      [else (iteration! p o n) (loop (add1 iterations))])))

(define seconds-per-cell 1.0)
(define repeats 3)

;; workers = 0 runs the loop on the main thread itself, the single-threaded
;; path that must not regress. Answers the rate and current-gc-milliseconds
;; (collector CPU time) per wall millisecond; a collection stops every
;; thread in the place.
(define (ops-per-second p o n workers)
  (define gc-before (current-gc-milliseconds))
  (define deadline (+ (current-inexact-monotonic-milliseconds)
                      (* 1000 seconds-per-cell)))
  (define-values (ms iterations)
    (cond
      [(zero? workers)
       (define start (current-inexact-monotonic-milliseconds))
       (define done (loop-until deadline p o n))
       (values (- (current-inexact-monotonic-milliseconds) start) (list done))]
      [else
       (run-parallel workers
                     (lambda (_i)
                       (at-set-num-threads 1)
                       (loop-until deadline p o n)))]))
  (cons (/ (* tensors-per-iteration (apply + iterations)) (/ ms 1000.0))
        (/ (- (current-gc-milliseconds) gc-before) ms)))

(define sizes '(8 64 256))
(define worker-counts '(0 1 2 4 8))

(module+ main
  (at-set-num-threads 1)
  (print-banner
   (format "Q5: tensors made per second (randn, add, matmul; n x n; ~a s cells, median of ~a; intra-op threads 1)"
           seconds-per-cell repeats)
   #:torch-version (shim-version))
  (print-table-header
   (append '("allocator" "n")
           (for/list ([w (in-list worker-counts)])
             (if (zero? w) "main thread" (format "~a workers" w)))
           '("8 workers / 1 worker" "GC ms per wall ms, 8 workers"
             "ledger bytes after" "load")))
  (for* ([v (in-list (workload-variants #:bare? #t))]
         [n (in-list sizes)])
    (define p (car v))
    (define o (cdr v))
    (iteration! p o n)
    (define cells
      (for/list ([w (in-list worker-counts)])
        (define samples
          (for/list ([_ (in-range repeats)]) (ops-per-second p o n w)))
        (cons (median (map car samples)) (median (map cdr samples)))))
    (define rates (map car cells))
    (print-table-row
     (append (list (prototype-name p) n)
             (map fmt-rate rates)
             (list (~r (/ (list-ref rates 4) (list-ref rates 1)) #:precision 2)
                   (~r (cdr (list-ref cells 4)) #:precision 2)
                   ((prototype-live-bytes p))
                   (load-average)))))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
