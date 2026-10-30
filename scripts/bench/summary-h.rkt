#lang racket/base

(require (only-in racket/format ~a)
         (only-in racket/list filter-map remove-duplicates)
         (only-in racket/match match-define)
         (only-in racket/string string-join)
         (only-in "markdown.rkt"
                  find-record host iqr median median-at median-of num percent ratio ref share
                  table whole))

(provide h-case?
         h-context
         h-section)

(define h-cases '("cifar-loader" "cifar-resnet"))

(define min-rounds 3)

(define (base-case c)
  (and (string? c) (regexp-replace #rx"^py-" c "")))

(define (h-case? c)
  (and (member (base-case c) h-cases) #t))

(define (h-context r)
  (format "~a; run ~a" (host r) (ref r 'run_id)))

(define (setting r key)
  (define v (ref r 'settings key))
  (and v (string->number (~a v))))

(define (spread r)
  (define m (median-of r))
  (define q (iqr r))
  (define n (ref r 'stats 'n))
  (and (real? m) (real? q) (positive? m) (real? n) (>= n min-rounds) (* 100.0 (/ q m))))

(define (floor-of rk py)
  (define spreads (list (spread rk) (spread py)))
  (and (andmap real? spreads) (apply max spreads)))

(define (load-note rk py)
  (define loads (filter real? (append (or (ref rk 'run_loads) '()) (or (ref py 'run_loads) '()))))
  (define limit (or (ref rk 'max_load) (ref py 'max_load)))
  (define peak (and (pair? loads) (apply max loads)))
  (and (real? limit) peak (>= peak limit) (format "load ~a ≥ ~a" (num peak 1) limit)))

(define (images-per-s r)
  (define rate (ref r 'steps_per_s 'median))
  (define batch (setting r 'BATCH))
  (and (real? rate) batch (* rate batch)))

(define (epoch-ms r)
  (define m (median-of r))
  (and (real? m) (* 1000.0 m)))

(define (h-row records key)
  (match-define (cons c variant) key)
  (define rk (find-record records #:case c #:impl "rktorch" #:variant variant))
  (define py (find-record records #:case (string-append "py-" c) #:impl "pytorch" #:variant variant))
  (define either (or rk py))
  (define rk-device (ref rk 'meta 'device))
  (define py-device (ref py 'meta 'device))
  (define paired? (and rk py (equal? rk-device py-device)))
  (define loader? (equal? c "cifar-loader"))
  (define (rate r)
    (cond
      [(not r) "-"]
      [loader? (whole (images-per-s r))]
      [else (num (median r) 2)]))
  (define pipeline (or (ref either 'settings 'PIPELINE) "-"))
  (define notes
    (filter values
            (list (and (not (and rk py)) "no twin")
                  (and rk py (not paired?) (format "device ~a≠~a" rk-device py-device))
                  (and (eq? (ref py 'meta 'pin_memory) #t) "pin_memory (PyTorch only)")
                  (and py (equal? pipeline "batched") "twin crops per image")
                  (load-note rk py))))
  (list (if loader? "loader only" "09 ResNet") pipeline (or (ref either 'settings 'WORKERS) "-")
        (or rk-device py-device "-") (if loader? "images/s" "s/epoch")
        (rate rk) (rate py) (if paired? (ratio (median-of rk) (median-of py)) "-")
        (percent (and paired? (floor-of rk py)))
        (num (ref rk 'steps_per_s 'median) 1) (num (ref py 'steps_per_s 'median) 1)
        (num (median-at rk 'host_ms) 2) (num (median-at py 'host_ms) 2)
        (num (median-at rk 'step_ms) 2) (num (median-at py 'step_ms) 2)
        (whole (median-at rk 'epoch_gc_ms)) (share (median-at rk 'epoch_gc_ms) (epoch-ms rk))
        (whole (median-at py 'epoch_gc_ms)) (share (median-at py 'epoch_gc_ms) (epoch-ms py))
        (percent (spread rk)) (percent (spread py))
        (if (null? notes) "-" (string-join notes "; "))))

(define (h-section records)
  (define keys
    (remove-duplicates (for/list ([r (in-list records)])
                         (cons (base-case (ref r 'case)) (ref r 'variant)))))
  (define rounds (filter-map (lambda (r) (ref r 'stats 'n)) records))
  (and (pair? keys)
       (string-append
        (table '("rows" "pipeline" "workers" "device" "unit" "rktorch" "PyTorch"
                 "time rktorch/PyTorch" "noise floor" "steps/s rktorch" "steps/s PyTorch"
                 "host ms/step rktorch" "host ms/step PyTorch" "settled ms/step rktorch"
                 "settled ms/step PyTorch" "GC ms/epoch rktorch" "GC share rktorch"
                 "Python cyclic GC ms/epoch, main process" "Python cyclic GC share, main process"
                 "spread rktorch" "spread PyTorch" "notes")
               (map (lambda (key) (h-row records key)) keys))
        (format (string-append "\n\nNoise floor: the wider of a row's two spreads (IQR over"
                               " median), each from ~a rounds or more (here n = ~a); a time"
                               " ratio within it of 1 is noise.")
                min-rounds (if (pair? rounds) (apply min rounds) "?")))))

(module+ test
  (require rackunit)

  (define (meta device) (hasheq 'host "lab" 'device device 'threads 8))

  (define (h c impl variant s rate step host gc
             #:iqr [q 0.0] #:n [n 3] #:pin [pin #f] #:device [device "cuda"]
             #:loads [loads '(1.0)] #:max-load [max-load 4])
    (hasheq 'suite "e2e" 'group "summary" 'case c 'impl impl 'variant variant 'run_id "r1"
            'unit "s/epoch" 'stats (hasheq 'median s 'iqr q 'n n)
            'steps_per_s (hasheq 'median rate) 'step_ms (list step) 'host_ms (list host)
            'epoch_gc_ms (list gc) 'run_loads loads 'max_load max-load
            'settings (hasheq 'PIPELINE (regexp-replace #rx"-[0-9]+$" variant "")
                              'WORKERS (regexp-replace #rx"^.*-" variant "") 'BATCH "128")
            'meta (if (equal? impl "pytorch") (hash-set (meta device) 'pin_memory pin) (meta device))))

  (define (row-of md prefix)
    (for/first ([line (in-list (regexp-split #rx"\n" md))]
                #:when (regexp-match? (regexp (string-append "^" (regexp-quote prefix))) line))
      line))

  (test-case "H names the twins' cases, and no others"
    (check-true (h-case? "cifar-loader"))
    (check-true (h-case? "py-cifar-resnet"))
    (check-false (h-case? "17-workers"))
    (check-false (h-case? "py-09-resnet")))

  (test-case "H gives a loader row in images per second and a ResNet row in seconds per epoch"
    (define md
      (h-section
       (list (h "cifar-loader" "rktorch" "per-sample-4" 20.0 19.5 50.0 30.0 2000.0 #:iqr 2.0)
             (h "py-cifar-loader" "pytorch" "per-sample-4" 5.0 78.0 12.0 4.0 50.0 #:iqr 0.25
                #:pin #t)
             (h "cifar-resnet" "rktorch" "batched-2" 30.0 13.0 75.0 20.0 3000.0 #:iqr 1.5)
             (h "py-cifar-resnet" "pytorch" "batched-2" 10.0 39.0 25.0 5.0 0.0))))
    (check-equal?
     (row-of md "| loader only")
     (string-append "| loader only | per-sample | 4 | cuda | images/s | 2496 | 9984 | 4.00 | 10.0%"
                    " | 19.5 | 78.0 | 30.00 | 4.00 | 50.00 | 12.00 | 2000 | 10.0% | 50 | 1.0%"
                    " | 10.0% | 5.0% | pin_memory (PyTorch only) |"))
    (check-equal?
     (row-of md "| 09 ResNet")
     (string-append "| 09 ResNet | batched | 2 | cuda | s/epoch | 30.00 | 10.00 | 3.00 | 5.0%"
                    " | 13.0 | 39.0 | 20.00 | 5.00 | 75.00 | 25.00 | 3000 | 10.0% | 0 | 0.0%"
                    " | 5.0% | 0.0% | twin crops per image |"))
    (check-regexp-match #rx"each from 3 rounds or more \\(here n = 3\\)" md))

  (test-case "H's host ms/step is the pre-settle host time, beside the settled interval"
    (define md (h-section (list (h "cifar-loader" "rktorch" "batched-0" 1.0 10.0 40.0 7.0 0.0))))
    (check-regexp-match #rx"\\| host ms/step rktorch \\| host ms/step PyTorch \\| settled ms/step rktorch \\|" md)
    (check-regexp-match #rx"\\| 10\\.0 \\| - \\| 7\\.00 \\| - \\| 40\\.00 \\| - \\|" md))

  (test-case "H heads PyTorch's GC columns as the main process's cyclic collector"
    (define md (h-section (list (h "py-cifar-loader" "pytorch" "batched-0" 1.0 10.0 4.0 3.0 9.0))))
    (check-regexp-match
     #rx"\\| GC share rktorch \\| Python cyclic GC ms/epoch, main process \\| Python cyclic GC share, main process \\|"
     md))

  (test-case "H puts a split pair under one heading and rows every variant, paired or not"
    (define rk (h "cifar-loader" "rktorch" "per-sample-2" 10.0 39.0 25.0 5.0 100.0))
    (define py (h "py-cifar-loader" "pytorch" "per-sample-2" 2.0 78.0 12.0 3.0 50.0
                  #:device "cpu"))
    (define lone (h "py-cifar-loader" "pytorch" "per-sample-8" 2.0 78.0 12.0 3.0 50.0))
    (check-equal? (h-context rk) (h-context py))
    (define md (h-section (list rk py lone)))
    (check-regexp-match
     #rx"^\\| loader only \\| per-sample \\| 2 \\| cuda \\| images/s \\| 4992 \\| 9984 \\| - \\| - \\|"
     (row-of md "| loader only | per-sample | 2"))
    (check-regexp-match #rx"\\| device cuda≠cpu \\|$" (row-of md "| loader only | per-sample | 2"))
    (check-regexp-match
     #rx"^\\| loader only \\| per-sample \\| 8 \\| cuda \\| images/s \\| - \\| 9984 \\| - \\| - \\|.*\\| no twin \\|$"
     (row-of md "| loader only | per-sample | 8")))

  (test-case "H gives no spread or noise floor from fewer than 3 rounds"
    (define md
      (h-section
       (list (h "cifar-resnet" "rktorch" "per-sample-0" 30.0 13.0 75.0 20.0 0.0 #:iqr 3.0 #:n 1)
             (h "py-cifar-resnet" "pytorch" "per-sample-0" 10.0 39.0 25.0 5.0 0.0 #:n 1))))
    (check-regexp-match
     #rx"^\\| 09 ResNet \\| per-sample \\| 0 \\| cuda \\| s/epoch \\| 30\\.00 \\| 10\\.00 \\| 3\\.00 \\| - \\|"
     (row-of md "| 09 ResNet"))
    (check-regexp-match #rx"\\| - \\| - \\| - \\|$" (row-of md "| 09 ResNet"))
    (check-regexp-match #rx"\\(here n = 1\\)" md))

  (test-case "H marks a row whose runs met the load limit"
    (define md
      (h-section
       (list (h "cifar-loader" "rktorch" "batched-8" 1.0 10.0 4.0 3.0 0.0 #:loads '(2.0 3.5))
             (h "py-cifar-loader" "pytorch" "batched-8" 1.0 10.0 4.0 3.0 0.0
                #:loads '(9.125 2.0)))))
    (check-regexp-match #rx"\\| twin crops per image; load 9\\.1 ≥ 4 \\|$" (row-of md "| loader only"))
    (check-regexp-match
     #rx"\\| twin crops per image \\|$"
     (row-of (h-section (list (h "cifar-loader" "rktorch" "batched-8" 1.0 10.0 4.0 3.0 0.0)
                              (h "py-cifar-loader" "pytorch" "batched-8" 1.0 10.0 4.0 3.0 0.0)))
             "| loader only"))))
