#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/math exact-round)
         (only-in "markdown.rkt"
                  host impl-of make-section median num ref rktorch? settings-text table))

(provide l0-sections)

(define (cases records)
  (remove-duplicates (map (lambda (r) (ref r 'case)) records)))

(define (find records #:case c #:impl impl #:variant [variant #f])
  (for/last ([r (in-list records)]
             #:when (and (equal? (ref r 'case) c)
                         (equal? (impl-of r) impl)
                         (or (not variant) (equal? (ref r 'variant) variant))))
    r))

(define (ratio a b [digits 2])
  (if (and (real? a) (real? b) (positive? b)) (num (/ a b) digits) "-"))

(define (median-of r) (and r (median r)))

(define (whole x) (if (real? x) (number->string (exact-round x)) "-"))

(define (list-median xs)
  (define v (if (list? xs) (sort (filter real? xs) <) '()))
  (define n (length v))
  (cond
    [(zero? n) #f]
    [(odd? n) (list-ref v (quotient n 2))]
    [else (/ (+ (list-ref v (sub1 (quotient n 2))) (list-ref v (quotient n 2))) 2.0)]))

(define (placement r)
  (format "~a; ~a, ~a threads~a" (host r) (ref r 'meta 'device) (ref r 'meta 'threads)
          (if (string? (ref r 'meta 'gpu)) (format ", ~a" (ref r 'meta 'gpu)) "")))

(define (us r) (if r (num (/ (median r) 1000.0) 2) "-"))

(define (twin-op-section records)
  (table '("op" "rktorch facade µs" "rktorch raw µs" "PyTorch µs" "facade/PyTorch" "raw/PyTorch")
         (for/list ([op (in-list (cases records))])
           (define facade (find records #:case op #:impl "rktorch" #:variant "facade"))
           (define raw (find records #:case op #:impl "rktorch" #:variant "raw"))
           (define py (find records #:case op #:impl "pytorch"))
           (list op (us facade) (us raw) (us py)
                 (ratio (median-of facade) (median-of py))
                 (ratio (median-of raw) (median-of py))))))

(define (scaling-section records)
  (define counts
    (sort (remove-duplicates (map (lambda (r) (string->number (ref r 'variant))) records)) <))
  (define (at state n impl)
    (find records #:case state #:impl impl #:variant (number->string n)))
  (table '("main" "threads" "rktorch ops/s" "rktorch ×1" "PyTorch ops/s" "PyTorch ×1"
           "rktorch/PyTorch" "main steps/s rktorch" "main steps/s PyTorch")
         (for*/list ([state (in-list (cases records))] [n (in-list counts)])
           (define rk (at state n "rktorch"))
           (define py (at state n "pytorch"))
           (define (speedup impl)
             (ratio (median-of (at state n impl)) (median-of (at state (car counts) impl))))
           (define (mains r) (num (list-median (ref r 'main_steps_per_s)) 1))
           (list state n (whole (median-of rk)) (speedup "rktorch") (whole (median-of py))
                 (speedup "pytorch") (ratio (median-of rk) (median-of py)) (mains rk) (mains py)))))

(define (twin-name c) (string-append "py-" c))

(define (paired-e2e records keep?)
  (for*/list ([r (in-list records)]
              #:when (and (rktorch? r) (keep? (ref r 'case)))
              [py (in-value (find records #:case (twin-name (ref r 'case)) #:impl "pytorch"
                                  #:variant (ref r 'variant)))]
              #:when py)
    (cons r py)))

(define ((twin-e2e-section keep?) records)
  (define pairs (paired-e2e records keep?))
  (and (pair? pairs)
       (table '("example" "variant" "unit" "rktorch" "PyTorch" "rktorch/PyTorch"
                "step ms rktorch" "step ms PyTorch" "step ratio")
              (for/list ([p (in-list pairs)])
                (define rk (car p))
                (define py (cdr p))
                (define (step r) (list-median (ref r 'step_ms)))
                (list (ref rk 'case) (or (ref rk 'variant) "-") (ref rk 'unit)
                      (num (median rk) 2) (num (median py) 2) (ratio (median rk) (median py))
                      (num (step rk) 2) (num (step py) 2) (ratio (step rk) (step py)))))))

(define (loader-example? c) (equal? c "17-workers"))

(define (calls-section records)
  (table '("example" "phase" "native crossings" "accounted tensors" "settings")
         (for/list ([r (in-list records)])
           (list (ref r 'case) (ref r 'variant) (whole (median r))
                 (whole (ref r 'tensor_stats 'median)) (settings-text (ref r 'settings))))))

(define (host-gpu-section records)
  (define (at group variant)
    (median-of (for/last ([r (in-list records)]
                          #:when (and (equal? (ref r 'group) group) (equal? (ref r 'variant) variant)))
                 r)))
  (define phases
    (remove-duplicates
     (for*/list ([r (in-list records)]
                 #:when (equal? (ref r 'group) "phase")
                 [m (in-value (regexp-match #rx"^[^/]*/(.*)$" (ref r 'variant)))])
       (cadr m))))
  (define (phase mode name) (at "phase" (format "~a/~a" mode name)))
  (define (enqueue mode)
    (for/sum ([name (in-list phases)] #:unless (equal? name "loss-item"))
      (or (phase mode name) 0)))
  (define (ms x) (num x 2))
  (table '("phase" "host ms, item every step" "host ms, no item" "ms, synced per phase")
         (append
          (for/list ([name (in-list phases)])
            (list name (ms (phase 'item name)) (ms (phase 'no-item name))
                  (ms (phase 'phase-sync name))))
          (list (list "host enqueue (no loss-item)" (ms (enqueue 'item)) (ms (enqueue 'no-item))
                      (ms (enqueue 'phase-sync)))
                (list "step" (ms (at "step" "item")) (ms (at "step" "no-item"))
                      (ms (at "step" "phase-sync")))
                (list "host enqueue / step" (ratio (enqueue 'item) (at "step" "item"))
                      (ratio (enqueue 'no-item) (at "step" "no-item")) "-")))))

(define (gc-section records)
  (define rows
    (for/list ([r (in-list records)] #:when (pair? (ref r 'epoch_gc_ms)))
      (define gc (list-median (ref r 'epoch_gc_ms)))
      (define epoch-ms (and (equal? (ref r 'unit) "s/epoch") (* 1000.0 (median r))))
      (list (ref r 'case) (or (ref r 'variant) "-") (settings-text (ref r 'settings)) (ref r 'unit)
            (num (median r) 2) (num (ref r 'steps_per_s 'median) 2) (whole gc)
            (if (and gc epoch-ms) (format "~a%" (num (* 100 (/ gc epoch-ms)) 1)) "-"))))
  (and (pair? rows)
       (table '("example" "variant" "settings" "unit" "median" "steps/s" "GC ms/epoch" "GC share")
              rows)))

(define l0-sections
  (list (make-section "micro" "ops" "A. Per op against PyTorch" twin-op-section placement)
        (make-section "scaling" "threads" "B. Thread scaling" scaling-section host)
        (make-section "e2e" "summary" "C. Loader end to end against PyTorch"
                      (twin-e2e-section loader-example?) host)
        (make-section "e2e" "summary" "D. Training against PyTorch"
                      (twin-e2e-section (lambda (c) (not (loader-example? c)))) host)
        (make-section "calls" "phase" "E. Native calls per step" calls-section placement)
        (make-section "host-gpu" #f "F. Host and device time per step" host-gpu-section placement)
        (make-section "e2e" "summary" "G. GC per epoch" gc-section host #:keep rktorch?)))

(module+ test
  (require rackunit
           (only-in "markdown.rkt" section-render))
  (define (render n records)
    ((section-render (list-ref l0-sections n)) records))
  (define meta (hasheq 'host "lab" 'device "cpu" 'threads 8))

  (test-case "A pairs each op's facade and raw with PyTorch"
    (define (op variant impl ns)
      (hasheq 'case "add-8x8" 'variant variant 'impl impl 'stats (hasheq 'median ns) 'meta meta))
    (define md (render 0 (list (op "facade" "rktorch" 14000.0) (op "raw" "rktorch" 9000.0)
                               (op "pytorch" "pytorch" 2000.0))))
    (check-regexp-match #rx"\\| add-8x8 \\| 14\\.00 \\| 9\\.00 \\| 2\\.00 \\| 7\\.00 \\| 4\\.50 \\|" md))

  (test-case "B gives each implementation's speedup over one thread, and their ratio"
    (define (row impl state n ops)
      (hasheq 'case state 'variant (number->string n) 'impl impl 'meta meta
              'stats (hasheq 'median ops) 'main_steps_per_s '(10.0 12.0)))
    (define md (render 1 (list (row "rktorch" "idle" 1 100.0) (row "rktorch" "idle" 2 185.0)
                               (row "pytorch" "idle" 1 200.0) (row "pytorch" "idle" 2 400.0))))
    (check-regexp-match #rx"\\| idle \\| 1 \\| 100 \\| 1\\.00 \\| 200 \\| 1\\.00 \\| 0\\.50 \\| 11\\.0 \\| 11\\.0 \\|" md)
    (check-regexp-match #rx"\\| idle \\| 2 \\| 185 \\| 1\\.85 \\| 400 \\| 2\\.00 \\| 0\\.46 \\|" md))

  (define (e2e c impl variant s step #:gc [gc '()] #:settings [settings (hasheq)])
    (hasheq 'suite "e2e" 'group "summary" 'case c 'impl impl 'variant variant 'unit "s/epoch"
            'stats (hasheq 'median s) 'step_ms (list step) 'epoch_gc_ms gc 'settings settings
            'steps_per_s (hasheq 'median 100.0) 'meta meta))

  (test-case "C and D pair an example with its twin, variant by variant"
    (define records (list (e2e "17-workers" "rktorch" "crop-w4" 25.0 50.0)
                          (e2e "py-17-workers" "pytorch" "crop-w4" 5.0 10.0)
                          (e2e "05-mnist" "rktorch" #f 1.3 2.6)
                          (e2e "py-05-mnist" "pytorch" #f 1.0 2.0)
                          (e2e "09-resnet" "rktorch" #f 14.0 30.0)))
    (define c (render 2 records))
    (check-regexp-match #rx"\\| 17-workers \\| crop-w4 \\| s/epoch \\| 25\\.00 \\| 5\\.00 \\| 5\\.00 \\| 50\\.00 \\| 10\\.00 \\| 5\\.00 \\|" c)
    (check-false (regexp-match? #rx"05-mnist" c))
    (define d (render 3 records))
    (check-regexp-match #rx"\\| 05-mnist \\| - \\| s/epoch \\| 1\\.30 \\| 1\\.00 \\| 1\\.30 \\|" d)
    (check-false (regexp-match? #rx"09-resnet|17-workers" d) "an example without a twin is left out")
    (check-false (render 3 (list (e2e "09-resnet" "rktorch" #f 14.0 30.0)))))

  (test-case "E lists crossings and tensors per phase"
    (check-regexp-match
     #rx"\\| 05-mnist \\| optimizer \\| 243 \\| 16 \\| batch=128 \\|"
     (render 4 (list (hasheq 'case "05-mnist" 'variant "optimizer" 'stats (hasheq 'median 243.0)
                             'tensor_stats (hasheq 'median 16.0) 'settings (hasheq 'batch 128)
                             'meta meta)))))

  (test-case "F sets host enqueue time beside the step it overlaps"
    (define (r group variant ms)
      (hasheq 'group group 'variant variant 'stats (hasheq 'median ms) 'meta meta))
    (define md
      (render 5 (list (r "phase" "item/forward" 4.0) (r "phase" "item/loss-item" 6.0)
                      (r "phase" "no-item/forward" 4.0) (r "phase" "phase-sync/forward" 9.0)
                      (r "step" "item" 10.0) (r "step" "no-item" 8.0) (r "step" "phase-sync" 12.0))))
    (check-regexp-match #rx"\\| forward \\| 4\\.00 \\| 4\\.00 \\| 9\\.00 \\|" md)
    (check-regexp-match #rx"\\| loss-item \\| 6\\.00 \\| - \\| - \\|" md)
    (check-regexp-match #rx"\\| step \\| 10\\.00 \\| 8\\.00 \\| 12\\.00 \\|" md)
    (check-regexp-match #rx"\\| host enqueue / step \\| 0\\.40 \\| 0\\.50 \\| - \\|" md))

  (test-case "G gives GC per epoch and its share of the epoch"
    (define md (render 6 (list (e2e "05-mnist" "rktorch" "no-troughs" 1.25 2.0 #:gc '(100 150)
                                    #:settings (hasheq 'RKTORCH_BENCH_COLLECT_AT_TROUGHS "0")))))
    (check-regexp-match
     #rx"\\| 05-mnist \\| no-troughs \\| RKTORCH_BENCH_COLLECT_AT_TROUGHS=0 \\| s/epoch \\| 1\\.25 \\| 100\\.00 \\| 125 \\| 10\\.0% \\|"
     md)
    (check-false (render 6 (list (e2e "05-mnist" "rktorch" #f 1.0 2.0))))))
