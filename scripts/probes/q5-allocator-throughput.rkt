#lang racket/base

(require (only-in racket/format ~a ~r)
         (only-in racket/function const)
         (only-in racket/match == match match-define)
         (only-in racket/math exact-round)
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
  (loop-for-seconds seconds-per-cell (lambda (_i) (iteration! p o n))))

(define repeats 3)
(define settle-round-limit 50)

(struct exn:fail:unsettled exn:fail ())

(define (settle! p when-settling)
  (or (settle-wills! #:limit settle-round-limit)
      (raise (exn:fail:unsettled
              (format "~a: wills still ran after ~a major collections ~a"
                      (prototype-name p) settle-round-limit when-settling)
              (current-continuation-marks)))))

(struct cell (workers rate gc-share made calls pending))

(define (measure-cell p o n workers mode)
  (settle! p (format "before n = ~a, ~a workers" n workers))
  (define calls (prototype-finalizer-calls p))
  (define window
    (run-parallel workers
                  (lambda (_i) (loop-for-cell p o n))
                  #:setup (lambda () (at-set-num-threads 1))
                  #:counter (or calls (const 0))
                  #:main mode))
  (define pending (settle! p (format "after n = ~a, ~a workers" n workers)))
  (define made (* tensors-per-iteration (apply + (span-value window))))
  (cell workers
        (/ made (ms->seconds (span-ms window)))
        (/ (span-gc-ms window) (span-ms window))
        made
        (and calls (span-counted window))
        pending))

(define (median-cell workers samples)
  (define (median-by field)
    (define xs (map field samples))
    (and (andmap values xs) (median-of xs)))
  (cell workers
        (median-by cell-rate)
        (median-by cell-gc-share)
        (median-by cell-made)
        (median-by cell-calls)
        (median-by cell-pending)))

(define (measured p o n workers mode)
  (median-cell workers (repeat-runs repeats (lambda () (measure-cell p o n workers mode)))))

(define sizes '(8 64 256))
(define worker-counts '(1 2 4 8))

(define (cell-for cells workers)
  (findf (lambda (c) (and c (= (cell-workers c) workers))) cells))

(define (count->string k)
  (match k
    [#f "uncounted"]
    [_ (~a (exact-round k))]))

(define (by-column cells show)
  (~> cells
      (map (lambda (c) (if c (show c) "-")) _)
      (string-join " / ")))

(define (allocator-rows p o n)
  (iteration! p o n)
  (define alone (measured p o n 0 main-blocked))
  (for ([mode (in-list (main-modes seconds-per-cell))])
    (define cells
      (cons (match mode
              [(== main-blocked eq?) alone]
              [_ #f])
            (for/list ([w (in-list worker-counts)]) (measured p o n w mode))))
    (define one (cell-for cells 1))
    (define eight (cell-for cells 8))
    (print-table-row
     (append (list (prototype-name p) n (main-mode-label mode))
             (map (lambda (c) (if c (rate->string (cell-rate c)) "-")) cells)
             (list (~r (/ (cell-rate eight) (cell-rate one)) #:precision 2)
                   (~r (cell-gc-share eight) #:precision 2)
                   (by-column cells (lambda (c) (count->string (cell-made c))))
                   (by-column cells (lambda (c) (count->string (cell-calls c))))
                   (by-column cells (lambda (c) (count->string (cell-pending c))))
                   ((prototype-live-bytes p))
                   (load-average))))))

(module+ main
  (at-set-num-threads 1)
  (print-banner
   (format "Q5: tensors made per second (randn, add, matmul; n x n; ~a s cells, median of ~a; intra-op threads 1)"
           seconds-per-cell repeats)
   #:torch-version (shim-version))
  (displayln
   (string-append
    "- every tensor is freed explicitly, so a will that runs later only finds it freed;"
    " wills pending at stop are counted for every allocator by running them on the main thread"
    " after each major collection until one collection leaves none;"
    " finalizer calls in the window are uncounted where the allocator's wills belong to"
    " ffi/unsafe/alloc (today, #:merely-uninterruptible?);"
    " rates divide by the whole window, which lasts until every worker exits\n"))
  (print-table-header
   (append '("allocator" "n" "main thread while workers run" "main thread runs the loop itself")
           (for/list ([w (in-list worker-counts)]) (format "~a workers" w))
           '("8 workers / 1 worker"
             "GC wall ms per wall ms (excludes rendezvous wait), 8 workers"
             "tensors made, by column"
             "finalizer calls in the window, by column"
             "wills pending at stop, by column"
             "ledger bytes after" "load")))
  (with-handlers ([exn:fail:unsettled?
                   (lambda (e)
                     (printf "\nstopped: ~a, so the cell would time another row's finalizers\n"
                             (exn-message e)))])
    (for ([make (in-list (variant-makers #:bare? #t))])
      (match-define (variant p o) (make))
      (for ([n (in-list sizes)])
        (allocator-rows p o n))
      (settle! p "after its last row, before the next allocator")))
  (print-footer))
