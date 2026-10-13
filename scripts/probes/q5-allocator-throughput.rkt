#lang racket/base

(require (only-in racket/format ~a ~r)
         (only-in racket/function const)
         (only-in racket/match match match-define)
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

(define (no-count p)
  (cond
    [(prototype-finalizer-frees p) "uncounted"]
    [else "n/a"]))

(define (measure-cell p o n workers mode)
  (settle! p (format "before n = ~a, ~a workers" n workers))
  (define calls (prototype-finalizer-calls p))
  (define t (tally workers))
  (define stop (box #f))
  (define (made-so-far) (* tensors-per-iteration (tally-total t)))
  (define calls-before (and calls (calls)))
  (define window
    (run-parallel workers
                  (lambda (i)
                    (tally-loop-for-seconds t i seconds-per-cell
                                            (lambda (_k) (iteration! p o n))
                                            #:stop? (lambda (_k) (unbox stop))))
                  #:setup (lambda () (at-set-num-threads 1))
                  #:counters (list made-so-far (or calls (const 0)))
                  #:main mode
                  #:on-end (lambda () (tally-close! t))
                  #:stop stop))
  (define calls-after (and calls (calls)))
  (settle! p (format "after n = ~a, ~a workers" n workers))
  (define calls-settled (and calls (calls)))
  (match-define (list made window-calls) (span-counts window))
  (define-values (counted-calls pending unaccounted)
    (cond
      [calls (values window-calls
                     (- calls-settled calls-after)
                     (- (made-so-far) (- calls-settled calls-before)))]
      [else (values (no-count p) (no-count p) (no-count p))]))
  (cell workers
        (/ made (ms->seconds (span-ms window)))
        (/ (span-gc-ms window) (span-ms window))
        (tally-max-gap t)
        made
        counted-calls
        pending
        unaccounted))

(define (median-cell workers samples)
  (define (median-by field)
    (match (map field samples)
      [(list (? real? xs) ...) (median-of xs)]
      [(cons marker _) marker]))
  (define middle (middle-by cell-made samples))
  (define middle-gap (middle-by (lambda (c) (gap-ms (cell-longest-gap c))) samples))
  (cell workers
        (median-by cell-rate)
        (median-by cell-gc-share)
        (cell-longest-gap middle-gap)
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
  (cond
    [(real? k) (~a (exact-round k))]
    [else k]))

(define ((or-dash show) c)
  (if c (show c) "-"))

(define (by-column cells show)
  (~> cells
      (map (or-dash show) _)
      (string-join " / ")))

(define (count-column cells field)
  (by-column cells (lambda (c) (count->string (field c)))))

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
                   (by-column cells (lambda (c) (gap->string (cell-longest-gap c))))
                   (count-column cells cell-made)
                   (count-column cells cell-calls)
                   (count-column cells cell-pending)
                   (count-column cells cell-unaccounted)
                   ((prototype-live-bytes p))
                   (load-average))))))

(module+ main
  (at-set-num-threads 1)
  (print-banner
   (format "Q5: allocator throughput, tensors/s (~a s cells, median of ~a)" seconds-per-cell repeats)
   #:torch-version (shim-version))
  (print-table-header
   (append '("allocator" "n" "main thread" "main alone")
           (for/list ([w (in-list worker-counts)]) (format "~a workers" w))
           '("8 / 1 workers" "GC share, 8 workers" "longest gap ms" "made" "finalizer calls"
             "pending at stop" "unaccounted" "ledger bytes after" "load")))
  (with-handlers ([exn:fail:unsettled?
                   (lambda (e)
                     (printf "\nstopped: ~a\n" (exn-message e)))])
    (for ([make (in-list (variant-makers #:bare? #t))])
      (match-define (variant p o) (make))
      (for ([n (in-list sizes)])
        (allocator-rows p o n))
      (settle! p "after its last row, before the next allocator")))
  (print-footer))
