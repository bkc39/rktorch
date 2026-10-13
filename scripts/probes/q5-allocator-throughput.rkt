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
  (define h (prototype-handle p))
  (define free! (prototype-free! p))
  (match-define (ops randn add matmul) o)
  (define a (randn n))
  (define b (add (h a) (h a)))
  (define c (matmul (h b) (h a)))
  (free! c)
  (free! b)
  (free! a))

(define tensors-per-iteration 3)

(define seconds-per-cell 1.0)

(define (loop-for-cell p o n)
  (define deadline (+ (current-inexact-monotonic-milliseconds) (* 1000 seconds-per-cell)))
  (let loop ([iterations 0])
    (cond
      [(> (current-inexact-monotonic-milliseconds) deadline) iterations]
      [else (iteration! p o n) (loop (add1 iterations))])))

(define repeats 3)
(define settle-round-limit 50)
(define (settle-finalizers! p)
  (define calls (prototype-finalizer-calls p))
  (let loop ([rounds 1] [before (calls)] [quiet 0])
    (define canary-seen? (collect-past-canary!))
    (sleep 0.01)
    (define now (calls))
    (define quiet* (if (and canary-seen? (= now before)) (add1 quiet) 0))
    (cond
      [(= quiet* 2) #t]
      [(= rounds settle-round-limit) #f]
      [else (loop (add1 rounds) now quiet*)])))

(struct cell (workers rate gc-share frees pending settle-ms))

(struct unsettled (reason))

(define (settle-or-stop! p when-settling)
  (unless (settle-finalizers! p)
    (raise (unsettled (format "~a: finalizers did not settle in ~a major collections ~a"
                              (prototype-name p) settle-round-limit when-settling)))))

(define (measure-cell p o n workers mode)
  (settle-or-stop! p (format "before n = ~a, ~a workers" n workers))
  (define frees (prototype-finalizer-frees p))
  (define calls (prototype-finalizer-calls p))
  (define-values (ms iterations gc-ms finalizer-frees)
    (cond
      [(zero? workers)
       (at-set-num-threads 1)
       (define frees-start (frees))
       (define gc-start (current-gc-milliseconds))
       (define start (current-inexact-monotonic-milliseconds))
       (define done (loop-for-cell p o n))
       (define ms (- (current-inexact-monotonic-milliseconds) start))
       (define gc-ms (- (current-gc-milliseconds) gc-start))
       (values ms (list done) gc-ms (- (frees) frees-start))]
      [else
       (run-parallel/gc workers
                        (lambda (_i) (loop-for-cell p o n))
                        #:setup (lambda () (at-set-num-threads 1))
                        #:counter frees
                        #:main mode)]))
  (define calls-at-stop (calls))
  (define settle-start (current-inexact-monotonic-milliseconds))
  (settle-or-stop! p (format "after n = ~a, ~a workers" n workers))
  (cell workers
        (/ (* tensors-per-iteration (apply + iterations)) (/ ms 1000.0))
        (/ gc-ms ms)
        finalizer-frees
        (- (calls) calls-at-stop)
        (- (current-inexact-monotonic-milliseconds) settle-start)))

(define (median-cell samples)
  (cell (cell-workers (car samples))
        (median < (map cell-rate samples))
        (median < (map cell-gc-share samples))
        (median < (map cell-frees samples))
        (median < (map cell-pending samples))
        (median < (map cell-settle-ms samples))))

(define sizes '(8 64 256))
(define worker-counts '(1 2 4 8))

(define (cell-for cells workers)
  (findf (lambda (c) (and c (= (cell-workers c) workers))) cells))

(define (by-column cells show)
  (string-join (map (lambda (c) (if c (show c) "-")) cells) " / "))

(define (measured p o n workers mode)
  (median-cell (for/list ([_ (in-range repeats)]) (measure-cell p o n workers mode))))

(module+ main
  (at-set-num-threads 1)
  (print-banner
   (format "Q5: tensors made per second (randn, add, matmul; n x n; ~a s cells, median of ~a; intra-op threads 1)"
           seconds-per-cell repeats)
   #:torch-version (shim-version))
  (print-table-header
   (append '("allocator" "n" "main thread while workers run" "main thread runs the loop itself")
           (for/list ([w (in-list worker-counts)]) (format "~a workers" w))
           '("8 workers / 1 worker" "GC ms per wall ms, 8 workers"
             "finalizer frees per cell, by column"
             "allocator finalizer calls still pending at stop, by column"
             "ms to settle after stop, by column"
             "ledger bytes after" "load")))
  (with-handlers ([unsettled?
                   (lambda (e)
                     (printf "\nstopped: ~a, so the cell would time another row's finalizers\n"
                             (unsettled-reason e)))])
    (for ([make (in-list (variant-makers #:bare? #t))])
      (match-define (variant p o) (make))
      (for ([n (in-list sizes)])
        (iteration! p o n)
        (define alone (measured p o n 0 main-busy))
        (for ([mode (in-list main-modes)]
              [first? (in-sequences (list #t) (in-cycle (list #f)))])
          (define cells
            (cons (and first? alone)
                  (for/list ([w (in-list worker-counts)]) (measured p o n w mode))))
          (define one (cell-for cells 1))
          (define eight (cell-for cells 8))
          (print-table-row
           (append (list (prototype-name p) n (main-mode-label mode))
                   (map (lambda (c) (if c (fmt-rate (cell-rate c)) "-")) cells)
                   (list (~r (/ (cell-rate eight) (cell-rate one)) #:precision 2)
                         (~r (cell-gc-share eight) #:precision 2)
                         (by-column cells (lambda (c) (fmt-rate (cell-frees c))))
                         (by-column cells (lambda (c) (fmt-rate (cell-pending c))))
                         (by-column cells (lambda (c) (fmt-ms (cell-settle-ms c))))
                         ((prototype-live-bytes p))
                         (load-average))))))
      (settle-or-stop! p "after its last row, before the next allocator")))
  (printf "\nload average (1 min) at end: ~a\n" (load-average)))
