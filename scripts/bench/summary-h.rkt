#lang racket/base

(require (only-in racket/format ~a)
         (only-in racket/list filter-map)
         (only-in racket/math exact-round)
         (only-in threading ~>)
         (only-in "harness.rkt" list-median)
         (only-in "markdown.rkt" iqr median num ref rktorch? table))

(provide h-case?
         h-section)

(define h-cases '("cifar-loader" "cifar-resnet"))

(define (rktorch-case c)
  (and (string? c) (regexp-replace #rx"^py-" c "")))

(define (h-case? c)
  (and (member (rktorch-case c) h-cases) #t))

(define (loader-only? r) (equal? (rktorch-case (ref r 'case)) "cifar-loader"))

(define (setting r key)
  (define v (ref r 'settings key))
  (and v (string->number (~a v))))

(define (throughput r)
  (define rate (ref r 'steps_per_s 'median))
  (define batch (setting r 'BATCH))
  (cond
    [(not (loader-only? r)) (median r)]
    [(and (real? rate) batch) (* rate batch)]
    [else #f]))

(define (step-ms r) (and r (list-median (ref r 'step_ms))))

(define (gc-ms r) (and r (list-median (ref r 'epoch_gc_ms))))

(define (gc-share r)
  (define gc (gc-ms r))
  (define epoch (and r (median r)))
  (if (and (real? gc) (real? epoch) (positive? epoch))
      (format "~a%" (~> gc (/ (* 1000.0 epoch)) (* 100) (num 1)))
      "-"))

(define (spread r)
  (define m (and r (median r)))
  (define q (and r (iqr r)))
  (and (real? m) (real? q) (positive? m) (* 100.0 (/ q m))))

(define (whole x) (if (real? x) (number->string (exact-round x)) "-"))

(define (percent x) (if (real? x) (format "~a%" (num x 1)) "-"))

(define (twin-of records r)
  (for/last ([py (in-list records)]
             #:when (and (not (rktorch? py))
                         (equal? (ref py 'case) (string-append "py-" (ref r 'case)))
                         (equal? (ref py 'variant) (ref r 'variant))))
    py))

(define (time-ratio rk py)
  (define a (median rk))
  (define b (and py (median py)))
  (if (and (real? a) (real? b) (positive? b)) (num (/ a b) 2) "-"))

(define (unmatched py)
  (if (and py (eq? (ref py 'meta 'pin_memory) #t)) "pin_memory (PyTorch only)" "-"))

(define (h-section records)
  (define rows
    (for/list ([rk (in-list records)] #:when (rktorch? rk))
      (define py (twin-of records rk))
      (define (rate r)
        (define x (and r (throughput r)))
        (if (loader-only? rk) (whole x) (num x 2)))
      (list (if (loader-only? rk) "loader only" "09 ResNet")
            (or (ref rk 'settings 'PIPELINE) "-") (or (ref rk 'settings 'WORKERS) "-")
            (if (loader-only? rk) "images/s" "s/epoch")
            (rate rk) (rate py) (time-ratio rk py)
            (num (ref rk 'steps_per_s 'median) 1) (num (and py (ref py 'steps_per_s 'median)) 1)
            (num (step-ms rk) 2) (num (step-ms py) 2)
            (whole (gc-ms rk)) (gc-share rk) (whole (gc-ms py)) (gc-share py)
            (percent (spread rk)) (percent (spread py)) (unmatched py))))
  (define spreads (filter-map spread records))
  (define rounds (filter-map (lambda (r) (ref r 'stats 'n)) records))
  (and (pair? rows)
       (string-append
        (table '("rows" "pipeline" "workers" "unit" "rktorch" "PyTorch" "time rktorch/PyTorch"
                 "steps/s rktorch" "steps/s PyTorch" "host ms/step rktorch"
                 "host ms/step PyTorch" "GC ms/epoch rktorch" "GC share rktorch"
                 "GC ms/epoch PyTorch" "GC share PyTorch" "spread rktorch" "spread PyTorch"
                 "unmatched")
               rows)
        (format (string-append "\n\nNoise floor: the widest spread (IQR over median, n = ~a)"
                               " is ~a; a difference between two rows below it is noise.")
                (if (pair? rounds) (apply min rounds) "?")
                (percent (and (pair? spreads) (apply max spreads)))))))

(module+ test
  (require rackunit)

  (define meta (hasheq 'host "lab" 'device "cuda" 'threads 8))

  (define (h c impl variant s rate step gc #:iqr [q 0.0] #:pin [pin #f] #:batch [batch "128"])
    (hasheq 'suite "e2e" 'group "summary" 'case c 'impl impl 'variant variant 'run_id "r1"
            'unit "s/epoch" 'stats (hasheq 'median s 'iqr q 'n 3)
            'steps_per_s (hasheq 'median rate) 'step_ms (list step) 'epoch_gc_ms (list gc)
            'settings (hasheq 'PIPELINE (regexp-replace #rx"-[0-9]+$" variant "")
                              'WORKERS (regexp-replace #rx"^.*-" variant "") 'BATCH batch)
            'meta (if (equal? impl "pytorch") (hash-set meta 'pin_memory pin) meta)))

  (test-case "H names the twins' cases, and no others"
    (check-true (h-case? "cifar-loader"))
    (check-true (h-case? "py-cifar-resnet"))
    (check-false (h-case? "17-workers"))
    (check-false (h-case? "py-09-resnet")))

  (test-case "H gives a loader row in images per second and a ResNet row in seconds per epoch"
    (define md
      (h-section
       (list (h "cifar-loader" "rktorch" "per-sample-4" 20.0 19.5 50.0 2000.0 #:iqr 2.0)
             (h "py-cifar-loader" "pytorch" "per-sample-4" 5.0 78.0 12.0 50.0 #:iqr 0.25
                #:pin #t)
             (h "cifar-resnet" "rktorch" "batched-2" 30.0 13.0 75.0 3000.0 #:iqr 1.5)
             (h "py-cifar-resnet" "pytorch" "batched-2" 10.0 39.0 25.0 0.0))))
    (check-regexp-match
     #rx"\\| loader only \\| per-sample \\| 4 \\| images/s \\| 2496 \\| 9984 \\| 4\\.00 \\| 19\\.5 \\| 78\\.0 \\| 50\\.00 \\| 12\\.00 \\| 2000 \\| 10\\.0% \\| 50 \\| 1\\.0% \\| 10\\.0% \\| 5\\.0% \\| pin_memory \\(PyTorch only\\) \\|"
     md)
    (check-regexp-match
     #rx"\\| 09 ResNet \\| batched \\| 2 \\| s/epoch \\| 30\\.00 \\| 10\\.00 \\| 3\\.00 \\| 13\\.0 \\| 39\\.0 \\| 75\\.00 \\| 25\\.00 \\| 3000 \\| 10\\.0% \\| 0 \\| 0\\.0% \\| 5\\.0% \\| 0\\.0% \\| - \\|"
     md)
    (check-regexp-match #rx"Noise floor: the widest spread \\(IQR over median, n = 3\\) is 10\\.0%" md))

  (test-case "H leaves a rktorch row without a twin unpaired"
    (define md (h-section (list (h "cifar-loader" "rktorch" "batched-8" 10.0 39.0 25.0 100.0))))
    (check-regexp-match
     #rx"\\| loader only \\| batched \\| 8 \\| images/s \\| 4992 \\| - \\| - \\| 39\\.0 \\| - \\|" md)
    (check-false (h-section (list (h "py-cifar-loader" "pytorch" "batched-8" 1.0 1.0 1.0 0.0))))))
