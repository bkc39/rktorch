#lang racket/base

(require (only-in racket/match match)
         (only-in racket/os getpid)
         (only-in "harness.rkt" busy-cores load-average write-record))

(provide fresh-run-id
         record-writer
         wait-for-load)

(define (fresh-run-id)
  (format "~a-~a-~a" (current-milliseconds) (getpid) (random 1000000000)))

(define ((record-writer run-id out) record)
  (write-record (hash-set record 'run_id run-id) out))

(define (current-load)
  (or (busy-cores)
      (match (load-average)
        [(cons one-minute _) one-minute]
        [_ #f])))

(define (wait-for-load limit log)
  (let loop ()
    (define load (and limit (current-load)))
    (cond
      [(or (not load) (< load limit)) (void)]
      [else
       (fprintf log "load ~a, waiting for < ~a\n" load limit)
       (sleep 10)
       (loop)])))
