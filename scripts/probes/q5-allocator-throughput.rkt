#lang racket/base

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
    (define canary-seen? (collect-past-canary!))
    (sleep 0.01)
    (define now (runs))
    (define quiet* (if (and canary-seen? (= now before)) (add1 quiet) 0))
    (cond
      [(= quiet* 2) #t]
      [(= rounds settle-round-limit) #f]
      [else (loop (add1 rounds) now quiet*)])))

(struct cell (workers rate gc-share finalizer-runs))

(struct unsettled (reason))

(define (settle-or-stop! p when-settling)
  (unless (settle-finalizers! p)
    (raise (unsettled (format "~a: finalizers did not settle in ~a major collections ~a"
                              (prototype-name p) settle-round-limit when-settling)))))

(define (measure-cell p o n workers)
  (settle-or-stop! p (format "before n = ~a, ~a workers" n workers))
  (define runs (prototype-finalizer-runs p))
  (define deadline (+ (current-inexact-monotonic-milliseconds)
                      (* 1000 seconds-per-cell)))
  (define-values (ms iterations gc-ms finalizer-runs)
    (cond
      [(zero? workers)
       (at-set-num-threads 1)
       (define runs-start (runs))
       (define gc-start (current-gc-milliseconds))
       (define start (current-inexact-monotonic-milliseconds))
       (define done (loop-until deadline p o n))
       (define ms (- (current-inexact-monotonic-milliseconds) start))
       (define gc-ms (- (current-gc-milliseconds) gc-start))
       (values ms (list done) gc-ms (- (runs) runs-start))]
      [else
       (run-parallel/gc workers
                        (lambda (_i) (loop-until deadline p o n))
                        #:setup (lambda () (at-set-num-threads 1))
                        #:counter runs)]))
  (cell workers
        (/ (* tensors-per-iteration (apply + iterations)) (/ ms 1000.0))
        (/ gc-ms ms)
        finalizer-runs))

(define (median-cell samples)
  (cell (cell-workers (car samples))
        (median < (map cell-rate samples))
        (median < (map cell-gc-share samples))
        (median < (map cell-finalizer-runs samples))))

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
             "finalizer runs per cell, by column"
             "ledger bytes after" "load")))
  (with-handlers ([unsettled?
                   (lambda (e)
                     (printf "\nstopped: ~a, so the cell would time another row's finalizers\n"
                             (unsettled-reason e)))])
    (for ([make (in-list (variant-makers #:bare? #t))])
      (match-define (variant p o) (make))
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
                       (string-join (map (lambda (c) (fmt-rate (cell-finalizer-runs c))) cells)
                                    " / ")
                       ((prototype-live-bytes p))
                       (load-average)))))
      (settle-or-stop! p "after its last row, before the next allocator")))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
