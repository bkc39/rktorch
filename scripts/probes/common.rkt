#lang racket/base

(require (only-in ffi/unsafe _fun _uintptr get-ffi-obj)
         (only-in racket/date date->string date-display-format)
         (only-in racket/file file->string)
         (only-in racket/format ~a ~r)
         (only-in racket/future processor-count)
         (only-in racket/list first)
         (only-in racket/math exact-round)
         (only-in racket/string string-join string-split))

(provide pthread-self
         main-os-thread
         on-main-os-thread?
         os-thread-label
         load-average
         os-task-count
         print-banner
         print-table
         print-table-header
         print-table-row
         run-parallel
         median
         fmt-ms
         fmt-rate)

(define pthread-self (get-ffi-obj "pthread_self" #f (_fun -> _uintptr)))

(define main-os-thread (pthread-self))

(define (on-main-os-thread?)
  (= (pthread-self) main-os-thread))

(define (os-thread-label id)
  (if (= id main-os-thread) "main" "own"))

(define (load-average)
  (first (string-split (file->string "/proc/loadavg"))))

(define (os-task-count)
  (length (directory-list "/proc/self/task")))

(define (print-banner title #:torch-version [torch-version #f])
  (printf "## ~a\n\n" title)
  (printf "- Racket ~a [~a], ~a cores, ~a\n"
          (version) (system-type 'vm) (processor-count)
          (parameterize ([date-display-format 'iso-8601])
            (date->string (seconds->date (current-seconds)) #t)))
  (when torch-version
    (printf "- libtorch ~a\n" torch-version))
  (printf "- load average (1 min) at start: ~a\n\n" (load-average)))

(define (row->line cells)
  (string-join (map ~a cells) " | " #:before-first "| " #:after-last " |"))

(define (print-table-header header)
  (displayln (row->line header))
  (displayln (row->line (map (lambda (_) "---") header)))
  (flush-output))

(define (print-table-row row)
  (displayln (row->line row))
  (flush-output))

(define (print-table header rows)
  (print-table-header header)
  (for-each print-table-row rows)
  (newline))

;; Every worker waits on one gate, so the clock starts with all of them ready.
(define (run-parallel n body #:pool [pool 'own])
  (define ready (make-semaphore 0))
  (define gate (make-semaphore 0))
  (define results (make-vector n #f))
  (define workers
    (for/list ([i (in-range n)])
      (thread (lambda ()
                (semaphore-post ready)
                (semaphore-wait gate)
                (vector-set! results i (body i)))
              #:pool pool)))
  (for ([_ (in-range n)]) (semaphore-wait ready))
  (define start (current-inexact-monotonic-milliseconds))
  (for ([_ (in-range n)]) (semaphore-post gate))
  (for-each thread-wait workers)
  (values (- (current-inexact-monotonic-milliseconds) start)
          (vector->list results)))

(define (median xs)
  (define sorted (sort xs <))
  (define n (length sorted))
  (cond
    [(odd? n) (list-ref sorted (quotient n 2))]
    [else (/ (+ (list-ref sorted (sub1 (quotient n 2)))
                (list-ref sorted (quotient n 2)))
             2)]))

(define (fmt-ms ms)
  (~r ms #:precision 1))

(define (fmt-rate per-second)
  (~a (exact-round per-second)))
