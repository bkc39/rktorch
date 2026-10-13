#lang racket/base

(require (only-in racket/format ~a ~r)
         (only-in racket/function const)
         (only-in racket/match match-define)
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

(define repeats 3)
(define settle-round-limit 50)

(struct exn:fail:unsettled exn:fail ())

(define (settle! p when-settling)
  (define-values (ran _collections) (settle-wills! #:limit settle-round-limit))
  (or ran
      (raise (exn:fail:unsettled
              (format "~a: wills still ran after ~a major collections ~a"
                      (prototype-name p) settle-round-limit when-settling)
              (current-continuation-marks)))))

(struct cell (workers rate gc-share longest-gap made calls pending unaccounted))

(define (measure-cell p o n workers mode)
  (settle! p (format "before n = ~a, ~a workers" n workers))
  (define calls (prototype-finalizer-calls p))
  (define tally (make-tally workers))
  (define stop (box #f))
  (define (made-so-far) (* tensors-per-iteration (tally-total tally)))
  (define calls-before (and calls (calls)))
  (define window
    (run-parallel workers
                  (lambda (i)
                    (tally-loop-for-seconds tally i seconds-per-cell
                                            (lambda (_k) (iteration! p o n))
                                            #:stop? (lambda (_k) (unbox stop))))
                  #:setup (lambda () (at-set-num-threads 1))
                  #:counters (list made-so-far (or calls (const 0)))
                  #:main mode
                  #:stop stop))
  (define calls-after (and calls (calls)))
  (define pending (settle! p (format "after n = ~a, ~a workers" n workers)))
  (match-define (list made window-calls) (span-counts window))
  (cell workers
        (/ made (ms->seconds (span-ms window)))
        (/ (span-gc-ms window) (span-ms window))
        (tally-max-gap tally)
        made
        (and calls window-calls)
        pending
        (and calls (- (made-so-far) (- calls-after calls-before) pending))))

(define (median-cell workers samples)
  (define (median-by field)
    (define xs (map field samples))
    (and (andmap values xs) (median-of xs)))
  (define middle (middle-by cell-made samples))
  (cell workers
        (median-by cell-rate)
        (median-by cell-gc-share)
        (median-by cell-longest-gap)
        (cell-made middle)
        (cell-calls middle)
        (cell-pending middle)
        (median-by cell-unaccounted)))

(define (measured p o n workers mode)
  (median-cell workers (repeat-runs repeats (lambda () (measure-cell p o n workers mode)))))

(define sizes '(8 64 256))
(define worker-counts '(1 2 4 8))

(define (cell-for cells workers)
  (findf (lambda (c) (and c (= (cell-workers c) workers))) cells))

(define (count->string k)
  (if k (~a (exact-round k)) "uncounted"))

(define ((or-dash show) c)
  (if c (show c) "-"))

(define (by-column cells show)
  (~> cells
      (map (or-dash show) _)
      (string-join " / ")))

(define (scaling one eight mode)
  (cond
    [(not (main-mode-polls-scheduler? mode)) "-"]
    [(zero? (cell-rate one)) "-"]
    [else (~r (/ (cell-rate eight) (cell-rate one)) #:precision 2)]))

(define (allocator-rows p o n)
  (iteration! p o n)
  (define alone (measured p o n 0 main-blocked))
  (for ([mode (in-list (main-modes seconds-per-cell))])
    (define cells
      (cons (and (eq? mode main-blocked) alone)
            (for/list ([w (in-list worker-counts)]) (measured p o n w mode))))
    (define one (cell-for cells 1))
    (define eight (cell-for cells 8))
    (print-table-row
     (append (list (prototype-name p) n (main-mode-label mode))
             (map (or-dash (lambda (c) (rate->string (cell-rate c)))) cells)
             (list (scaling one eight mode)
                   (~r (cell-gc-share eight) #:precision 2)
                   (by-column cells (lambda (c) (real->decimal-string (cell-longest-gap c) 1)))
                   (by-column cells (lambda (c) (count->string (cell-made c))))
                   (by-column cells (lambda (c) (count->string (cell-calls c))))
                   (by-column cells (lambda (c) (count->string (cell-pending c))))
                   (by-column cells (lambda (c) (count->string (cell-unaccounted c))))
                   ((prototype-live-bytes p))
                   (load-average))))))

(module+ main
  (at-set-num-threads 1)
  (print-banner
   (format "Q5: tensors made per second of window (randn, add, matmul; n x n; ~a s cells, median of ~a; intra-op threads 1); the window ends when the workers exit, or in the foreign-call modes at main's last wake"
           seconds-per-cell repeats)
   #:torch-version (shim-version))
  (print-table-header
   (append '("allocator" "n" "main thread while workers run" "main thread runs the loop itself")
           (for/list ([w (in-list worker-counts)]) (format "~a workers" w))
           '("8 workers / 1 worker"
             "GC wall ms per wall ms (excludes rendezvous wait), 8 workers"
             "longest ms between a worker's iterations, by column"
             "tensors made in the window, by column"
             "finalizer calls in the window, by column"
             "wills pending at stop, by column"
             "made - calls - pending over the whole run (median of runs), by column"
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
