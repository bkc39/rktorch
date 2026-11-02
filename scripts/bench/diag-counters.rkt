#lang racket/base

;; Counters the tensor-scope diagnostics read around a timed window, shared by
;; diag-scope.rkt and scope-pipelines.rkt. The finalizer timing counters only
;; tick when RKTORCH_DIAG_SCOPE is set, to any arm.

(require (only-in racket/string string-trim)
         (only-in torch finalizer-diagnostics)
         (only-in torch/foreign/raw/collector finalizer-runs finalizer-timing)
         (only-in torch/foreign/raw/scope scope-diagnostics))

(provide clock-ms
         counter-delta
         counter-snapshot
         ledger-entries
         load-average
         median
         median-record
         peak-rss-kb)

(define (clock-ms) (current-inexact-monotonic-milliseconds))

(define (ledger-entries)
  (cdr (assq 'ledger-entries (finalizer-diagnostics))))

(define (peak-rss-kb)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (call-with-input-file "/proc/self/status"
      (lambda (in)
        (for/or ([line (in-lines in)])
          (define m (regexp-match #rx"^VmHWM:[ \t]*([0-9]+)" line))
          (and m (string->number (cadr m))))))))

(define (load-average)
  (with-handlers ([exn:fail? (lambda (_) "?")])
    (string-trim (call-with-input-file "/proc/loadavg" read-line))))

(struct snapshot (ms gc runs main-runs us scope))

(define (counter-snapshot)
  (define-values (main us) (finalizer-timing))
  (snapshot (clock-ms) (current-gc-milliseconds) (finalizer-runs) main us
            (scope-diagnostics)))

;; totals over the window between two snapshots
(define (counter-delta s0 s1)
  (append
   (list (cons 'elapsed-s (/ (- (snapshot-ms s1) (snapshot-ms s0)) 1000.0))
         (cons 'gc-ms (- (snapshot-gc s1) (snapshot-gc s0)))
         (cons 'finalizer-runs (- (snapshot-runs s1) (snapshot-runs s0)))
         (cons 'finalizer-main-runs (- (snapshot-main-runs s1) (snapshot-main-runs s0)))
         (cons 'finalizer-ms (/ (- (snapshot-us s1) (snapshot-us s0)) 1000.0)))
   (for/list ([(k v) (in-hash (snapshot-scope s1))])
     (cons k (- v (hash-ref (snapshot-scope s0) k))))))

(define (median xs)
  (define sorted (sort xs <))
  (list-ref sorted (quotient (length sorted) 2)))

(define (median-record records)
  (for/list ([k (in-list (map car (car records)))])
    (define vs (filter real? (map (lambda (r) (cdr (or (assq k r) (cons k #f)))) records)))
    (cons k (if (null? vs) #f (median vs)))))
