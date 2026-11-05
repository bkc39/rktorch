#lang racket/base

(require (only-in ffi/unsafe/atomic call-as-atomic)
         (only-in threading ~>)
         (only-in "../../foreign.rkt" native-memory-unaccounted native-memory-use)
         (only-in "../../foreign/raw/collector.rkt" drain-finalizers!)
         (only-in (submod "../../runtime/reaper.rkt" private) reaper-statistics reaper-threads))

(provide median-allocation)

(define runs 15)

(define (phantom-bytes)
  (for/sum ([entry (in-list (append (native-memory-use) (native-memory-unaccounted)))])
    (cdr entry)))

;; A reaper counts its park before it waits and its wake after.
(define (reapers-parked?)
  (define stats (reaper-statistics))
  (= (- (hash-ref stats 'parks) (hash-ref stats 'wakes)) (length (reaper-threads))))

(define (await-parked-reapers)
  (define deadline (+ (current-inexact-milliseconds) 2000))
  (let wait ()
    (unless (or (reapers-parked?) (> (current-inexact-milliseconds) deadline))
      (sleep 0)
      (wait))))

;; Earlier runs' garbage is released, and every reaper parked again, before
;; the window opens: the heap counts every thread's allocation, and a woken
;; reaper's is not this run's.
(define (racket-heap-allocation thunk)
  (collect-garbage 'minor)
  (drain-finalizers!)
  (await-parked-reapers)
  (define start
    (call-as-atomic
     (lambda ()
       (define phantom (phantom-bytes))
       (- (current-memory-use 'cumulative) phantom))))
  (call-with-values thunk void)
  (call-as-atomic
   (lambda ()
     (define heap (current-memory-use 'cumulative))
     (- heap (phantom-bytes) start))))

(define (median-allocation thunk)
  (call-with-values thunk void)
  (~> (for/list ([_ (in-range runs)])
        (racket-heap-allocation thunk))
      (sort <)
      (list-ref (quotient runs 2))))

(module+ test
  (require (only-in rackunit check-true)
           (only-in "../../foreign.rkt" zeros))

  (define kept '())
  (define (keeping-zeros . dims)
    (median-allocation (lambda () (set! kept (cons (apply zeros dims) kept)))))
  (define one-element (keeping-zeros 1))
  (define four-mib (keeping-zeros 1024 1024))
  (check-true (< (abs (- four-mib one-element)) 4096)
              (format "~a bytes keeping a 4 MiB tensor, ~a keeping one element"
                      four-mib one-element))
  (set! kept '())

  (define bytes-call (median-allocation (lambda () (make-bytes (* 1024 1024)))))
  (check-true (>= bytes-call (* 1024 1024))
              (format "~a bytes for a call that allocates 1 MiB" bytes-call)))
