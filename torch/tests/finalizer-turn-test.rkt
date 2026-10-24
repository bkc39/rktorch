#lang racket/base

(module+ test
  (require (only-in racket/match match)
           rackunit
           (only-in "../foreign.rkt" conv2d cpu-device native-memory-use randn)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait! finalizer-runs))

  (define calls 200)
  (define result-bytes (* 128 16 32 32 4))

  (define (cpu-bytes)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (major? event)
    (match (vector-ref event 2)
      [(? struct? info)
       (and (memq (vector-ref (struct->vector info) 1) '(major major-incremental))
            #t)]
      [_ #f]))

  (define (count-majors receiver)
    (for/sum ([event (in-producer (lambda () (sync/timeout 0 receiver)) #f)])
      (if (major? event) 1 0)))

  (test-case "a loop of 8 MiB convs releases each result within a collection (#377)"
    (define x (randn 128 3 32 32))
    (define w (randn 16 3 3 3))
    (define b (randn 16))
    (define receiver (make-log-receiver (current-logger) 'debug 'GC))
    (for ([_ (in-range 3)]) (collect-and-wait!))
    (count-majors receiver)
    (define base (cpu-bytes))
    (define runs-before (finalizer-runs))
    (for ([_ (in-range calls)])
      (conv2d x w #:bias b #:padding 1))
    (define majors (count-majors receiver))
    (define held (- (cpu-bytes) base))
    (define released (- (finalizer-runs) runs-before))
    (define summary
      (format "~a MiB still charged, ~a of ~a released, ~a major collections"
              (quotient held 1048576) released calls majors))
    (check-true (<= held (* 4 result-bytes)) summary)
    (check-true (>= released (- calls 4)) summary)
    (check-true (<= majors 1) summary)))
