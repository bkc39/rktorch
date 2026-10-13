#lang racket/base

(require (only-in racket/format ~r)
         (only-in racket/match match-define)
         (only-in racket/string string-join)
         (only-in threading ~>)
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
  (loop-for-seconds seconds-per-cell (lambda () (iteration! p o n))))

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

(define (settle-or-stop! p stop when-settling)
  (unless (settle-finalizers! p)
    (stop (format "~a: finalizers did not settle in ~a major collections ~a"
                  (prototype-name p) settle-round-limit when-settling))))

(struct cell (workers rate gc-share frees pending settle-ms))

(define (measure-cell p o n workers mode stop)
  (settle-or-stop! p stop (format "before n = ~a, ~a workers" n workers))
  (define frees (prototype-finalizer-frees p))
  (define window
    (cond
      [(zero? workers)
       (at-set-num-threads 1)
       (define alone (measure-span (lambda () (loop-for-cell p o n)) #:counter frees))
       (struct-copy span alone [value (list (span-value alone))])]
      [else
       (run-parallel/gc workers
                        (lambda (_i) (loop-for-cell p o n))
                        #:setup (lambda () (at-set-num-threads 1))
                        #:counter frees
                        #:main mode)]))
  (define settling
    (measure-span (lambda ()
                    (settle-or-stop! p stop (format "after n = ~a, ~a workers" n workers)))
                  #:counter (prototype-finalizer-calls p)))
  (cell workers
        (/ (* tensors-per-iteration (apply + (span-value window))) (/ (span-ms window) 1000.0))
        (/ (span-gc-ms window) (span-ms window))
        (span-counted window)
        (span-counted settling)
        (span-ms settling)))

(define (median-cell workers samples)
  (define (median-by field) (median-of (map field samples)))
  (cell workers
        (median-by cell-rate)
        (median-by cell-gc-share)
        (median-by cell-frees)
        (median-by cell-pending)
        (median-by cell-settle-ms)))

(define (measured p o n workers mode stop)
  (median-cell workers (repeat-runs repeats (lambda () (measure-cell p o n workers mode stop)))))

(define sizes '(8 64 256))
(define worker-counts '(1 2 4 8))

(define (cell-for cells workers)
  (findf (lambda (c) (and c (= (cell-workers c) workers))) cells))

(define (by-column cells show)
  (~> cells
      (map (lambda (c) (if c (show c) "-")) _)
      (string-join " / ")))

(define (allocator-rows p o n stop)
  (iteration! p o n)
  (define alone (measured p o n 0 main-blocked stop))
  (for ([mode (in-list main-modes)]
        [i (in-naturals)])
    (define cells
      (cons (and (zero? i) alone)
            (for/list ([w (in-list worker-counts)]) (measured p o n w mode stop))))
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
  (define stopped
    (let/ec stop
      (for ([make (in-list (variant-makers #:bare? #t))])
        (match-define (variant p o) (make))
        (for ([n (in-list sizes)])
          (allocator-rows p o n stop))
        (settle-or-stop! p stop "after its last row, before the next allocator"))
      #f))
  (when stopped
    (printf "\nstopped: ~a, so the cell would time another row's finalizers\n" stopped))
  (print-footer))
