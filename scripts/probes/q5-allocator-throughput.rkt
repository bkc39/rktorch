#lang racket/base

;; Q5: tensors made per second from N parallel workers, today's allocator against the prototypes.

(require (only-in math/statistics median)
         (only-in racket/format ~r)
         (only-in racket/match match-define)
         (only-in racket/string string-join)
         "common.rkt"
         "prototype-allocator.rkt"
         "shim.rkt"
         "variants.rkt")

(define (iteration! p o n)
  (match-define (prototype _ _ h free! _ _) p)
  (match-define (ops randn add matmul) o)
  (define a (randn n))
  (define b (add (h a) (h a)))
  (define c (matmul (h b) (h a)))
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
(define settle-round-limit 50)

(define (settle-finalizers! p)
  (define runs (prototype-finalizer-runs p))
  (let loop ([rounds 1] [before (runs)] [quiet 0])
    (collect-garbage 'major)
    (sleep 0.01)
    (define now (runs))
    (define quiet* (if (= now before) (add1 quiet) 0))
    (cond
      [(= quiet* 2) #t]
      [(= rounds settle-round-limit) #f]
      [else (loop (add1 rounds) now quiet*)])))

(struct cell (workers rate gc-share finalizer-runs settled?))

;; workers = 0 runs the loop on the main thread itself, the single-threaded
;; path that must not regress. gc-share is current-gc-milliseconds (collector
;; CPU time) per wall millisecond; a collection stops every thread in the
;; place.
(define (measure-cell p o n workers)
  (define settled? (settle-finalizers! p))
  (define runs-before ((prototype-finalizer-runs p)))
  (define deadline (+ (current-inexact-monotonic-milliseconds)
                      (* 1000 seconds-per-cell)))
  (define-values (ms iterations gc-ms)
    (cond
      [(zero? workers)
       (define gc-start (current-gc-milliseconds))
       (define start (current-inexact-monotonic-milliseconds))
       (define done (loop-until deadline p o n))
       (values (- (current-inexact-monotonic-milliseconds) start)
               (list done)
               (- (current-gc-milliseconds) gc-start))]
      [else
       (run-parallel/gc workers
                        (lambda (_i)
                          (at-set-num-threads 1)
                          (loop-until deadline p o n)))]))
  (cell workers
        (/ (* tensors-per-iteration (apply + iterations)) (/ ms 1000.0))
        (/ gc-ms ms)
        (- ((prototype-finalizer-runs p)) runs-before)
        settled?))

(define (median-cell samples)
  (cell (cell-workers (car samples))
        (median < (map cell-rate samples))
        (median < (map cell-gc-share samples))
        (median < (map cell-finalizer-runs samples))
        (andmap cell-settled? samples)))

(define sizes '(8 64 256))
(define worker-counts '(0 1 2 4 8))

(define (cell-for cells workers)
  (findf (lambda (c) (= (cell-workers c) workers)) cells))

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
             "finalizer runs per cell, by column" "finalizers settled before every cell"
             "ledger bytes after" "load")))
  (for ([make (in-list (variant-makers #:bare? #t))])
    (match-define (variant p o _) (make))
    (for ([n (in-list sizes)])
      (iteration! p o n)
      (define cells
        (for/list ([w (in-list worker-counts)])
          (median-cell (for/list ([_ (in-range repeats)]) (measure-cell p o n w)))))
      (define one (cell-for cells 1))
      (define eight (cell-for cells 8))
      (print-table-row
       (append (list (prototype-name p) n)
               (map (lambda (c) (fmt-rate (cell-rate c))) cells)
               (list (~r (/ (cell-rate eight) (cell-rate one)) #:precision 2)
                     (~r (cell-gc-share eight) #:precision 2)
                     (string-join (map (lambda (c) (fmt-rate (cell-finalizer-runs c))) cells) " / ")
                     (if (andmap cell-settled? cells) "yes" "NO")
                     ((prototype-live-bytes p))
                     (load-average))))))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
