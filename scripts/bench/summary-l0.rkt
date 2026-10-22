#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/string string-join)
         (only-in racket/math exact-round)
         (only-in threading ~>)
         (only-in "harness.rkt" list-median)
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

(define (placement r)
  (format "~a; ~a, ~a threads~a" (host r) (ref r 'meta 'device) (ref r 'meta 'threads)
          (if (string? (ref r 'meta 'gpu)) (format ", ~a" (ref r 'meta 'gpu)) "")))

(define (us r) (if r (~> (median r) (/ 1000.0) (num 2)) "-"))

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
    (~> (map (lambda (r) (string->number (ref r 'variant))) records) remove-duplicates (sort <)))
  (define (at state n impl)
    (find records #:case state #:impl impl #:variant (number->string n)))
  (table '("main" "threads" "rktorch ops/s" "rktorch ÷ idle×1" "PyTorch ops/s" "PyTorch ÷ idle×1"
           "rktorch/PyTorch" "main steps/s rktorch" "main steps/s PyTorch")
         (for*/list ([state (in-list (cases records))] [n (in-list counts)])
           (define rk (at state n "rktorch"))
           (define py (at state n "pytorch"))
           (define (speedup impl)
             (ratio (median-of (at state n impl)) (median-of (at "idle" (car counts) impl))))
           (define (mains r) (~> (ref r 'main_steps_per_s) list-median (num 1)))
           (list state n (whole (median-of rk)) (speedup "rktorch") (whole (median-of py))
                 (speedup "pytorch") (ratio (median-of rk) (median-of py)) (mains rk) (mains py)))))

(define (twin-name c [suffix ""]) (string-append "py-" c suffix))

(define (paired-e2e records keep?)
  (for*/list ([r (in-list records)]
              #:when (and (rktorch? r) (keep? (ref r 'case)))
              [py (in-value (find records #:case (twin-name (ref r 'case)) #:impl "pytorch"
                                  #:variant (ref r 'variant)))]
              #:when py)
    (list r py (find records #:case (twin-name (ref r 'case) "-per-param") #:impl "pytorch"
                     #:variant (ref r 'variant)))))

(define (unmatched-settings rk py)
  (define rk-reads (ref rk 'reads))
  (define py-reads (ref py 'reads))
  (define (reads? names k) (and (member k names) #t))
  (cond
    [(and (list? rk-reads) (list? py-reads))
     (for/list ([k (in-list (~> (or (ref rk 'settings) (hasheq)) hash-keys
                                (map symbol->string _) (sort string<?)))]
                #:unless (eq? (reads? rk-reads k) (reads? py-reads k)))
       k)]
    [else '()]))

(define ((twin-e2e-section keep?) records)
  (define pairs (paired-e2e records keep?))
  (and (pair? pairs)
       (table '("example" "variant" "unit" "rktorch" "PyTorch" "rktorch/PyTorch"
                "PyTorch per-param" "rktorch/per-param" "step ms rktorch" "step ms PyTorch"
                "step ratio" "step ms per-param" "step ratio per-param" "unmatched settings")
              (for/list ([p (in-list pairs)])
                (match-define (list rk py pp) p)
                (define unmatched (unmatched-settings rk py))
                (define (step r) (and r (list-median (ref r 'step_ms))))
                (define (vs a b) (if (null? unmatched) (ratio a b) "-"))
                (list (ref rk 'case) (or (ref rk 'variant) "-") (ref rk 'unit)
                      (num (median rk) 2) (num (median py) 2) (vs (median rk) (median py))
                      (num (median-of pp) 2) (vs (median rk) (median-of pp))
                      (num (step rk) 2) (num (step py) 2) (vs (step rk) (step py))
                      (num (step pp) 2) (vs (step rk) (step pp))
                      (if (null? unmatched) "-" (string-join unmatched " ")))))))

(define (loader-example? c) (equal? c "17-workers"))

(define (calls-section records)
  (table '("example" "phase" "native crossings" "frees" "accounted tensors" "op calls" "settings")
         (for/list ([r (in-list records)])
           (list (ref r 'case) (ref r 'variant) (whole (median r))
                 (whole (ref r 'free_stats 'median)) (whole (ref r 'tensor_stats 'median))
                 (whole (ref r 'op_call_stats 'median)) (settings-text (ref r 'settings))))))

(define (host-gpu-section records)
  (define (record-at group variant)
    (for/last ([r (in-list records)]
               #:when (and (equal? (ref r 'group) group) (equal? (ref r 'variant) variant)))
      r))
  (define (at group variant) (median-of (record-at group variant)))
  (define phases
    (remove-duplicates
     (for*/list ([r (in-list records)]
                 #:when (equal? (ref r 'group) "phase")
                 [name (in-value (match (regexp-match #rx"^[^/]*/(.*)$" (ref r 'variant))
                                   [(list _ name) name]))])
       name)))
  (define (phase mode name) (at "phase" (format "~a/~a" mode name)))
  (define (enqueue mode)
    (for/sum ([name (in-list phases)] #:unless (equal? name "loss-item"))
      (or (phase mode name) 0)))
  (define (ms x) (num x 2))
  (define (gc variant) (list-median (ref (record-at "step" variant) 'gc_ms)))
  (define (net variant)
    (define r (record-at "step" variant))
    (define samples (ref r 'samples))
    (define gcs (ref r 'gc_ms))
    (and (list? samples) (list? gcs) (= (length samples) (length gcs))
         (list-median (map - samples gcs))))
  (define proxy (at "step" "proxy"))
  (define proxy-base (ref (record-at "step" "proxy") 'settings 'base))
  (table '("phase" "host ms, item every step" "host ms, no item" "ms, synced per phase")
         (append
          (for/list ([name (in-list phases)])
            (list name (ms (phase 'item name)) (ms (phase 'no-item name))
                  (ms (phase 'phase-sync name))))
          (list (list "phases without loss-item" (ms (enqueue 'item)) (ms (enqueue 'no-item))
                      (ms (enqueue 'phase-sync)))
                (list "step" (ms (at "step" "item")) (ms (at "step" "no-item"))
                      (ms (at "step" "phase-sync")))
                (list "GC in the step" (ms (gc "item")) (ms (gc "no-item")) (ms (gc "phase-sync")))
                (list "step net of GC" (ms (net "item")) (ms (net "no-item"))
                      (ms (net "phase-sync")))
                (list (format "host proxy: the step at base ~a" (or proxy-base "?"))
                      (ms proxy) "-" "-")
                (list "GC in the host proxy" (ms (gc "proxy")) "-" "-")
                (list "host proxy net of GC" (ms (net "proxy")) "-" "-")
                (list "host proxy / step" (ratio proxy (at "step" "item")) "-" "-")
                (list "host proxy / step, net of GC" (ratio (net "proxy") (net "item")) "-" "-")
                (list "device excess: the loss-item wait" (ms (phase 'item "loss-item")) "-" "-")))))

(define (gc-section records)
  (define rows
    (for/list ([r (in-list records)] #:when (pair? (ref r 'epoch_gc_ms)))
      (define gc (list-median (ref r 'epoch_gc_ms)))
      (define epoch-ms (and (equal? (ref r 'unit) "s/epoch") (* 1000.0 (median r))))
      (list (ref r 'case) (or (ref r 'variant) "-") (settings-text (ref r 'settings)) (ref r 'unit)
            (num (median r) 2) (num (ref r 'steps_per_s 'median) 2) (whole gc)
            (if (and gc epoch-ms) (format "~a%" (~> (/ gc epoch-ms) (* 100) (num 1))) "-"))))
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
           (only-in "markdown.rkt" by-context section-context section-render)
           (only-in "pytorch.rkt" twin-record))
  (define (render n records)
    ((section-render (list-ref l0-sections n)) records))
  (define meta (hasheq 'host "lab" 'device "cpu" 'threads 8))

  (test-case "A pairs each op's facade and raw with PyTorch"
    (define (op variant impl ns)
      (hasheq 'case "add-8x8" 'variant variant 'impl impl 'stats (hasheq 'median ns) 'meta meta))
    (define md (render 0 (list (op "facade" "rktorch" 14000.0) (op "raw" "rktorch" 9000.0)
                               (op "pytorch" "pytorch" 2000.0))))
    (check-regexp-match #rx"\\| add-8x8 \\| 14\\.00 \\| 9\\.00 \\| 2\\.00 \\| 7\\.00 \\| 4\\.50 \\|" md))

  (test-case "A pairs a CUDA twin with rktorch when only torch names the GPU"
    (define cuda (hasheq 'host "lab" 'device "cuda" 'threads 8 'gpu #f))
    (define (rk variant ns)
      (hasheq 'case "add-8x8" 'variant variant 'impl "rktorch" 'stats (hasheq 'median ns)
              'meta cuda))
    (define py
      (twin-record (hasheq 'case "add-8x8" 'variant "pytorch" 'samples '(2000.0)
                           'meta (hasheq 'impl "pytorch" 'device "cuda" 'threads 8
                                         'gpu "NVIDIA GeForce RTX 3090 Ti"))
                   #:suite "micro" #:meta cuda #:load '(0.0)))
    (define records (list (rk "facade" 14000.0) (rk "raw" 9000.0) py))
    (define groups (by-context records (section-context (car l0-sections))))
    (check-equal? (length groups) 1 "one table for the device, not one per implementation")
    (check-regexp-match #rx"\\| 7\\.00 \\| 4\\.50 \\|" (render 0 (cdr (car groups)))))

  (test-case "B scales each implementation against its own idle main on one thread"
    (define (row impl state n ops)
      (hasheq 'case state 'variant (number->string n) 'impl impl 'meta meta
              'stats (hasheq 'median ops) 'main_steps_per_s '(10.0 12.0)))
    (define md (render 1 (list (row "rktorch" "idle" 1 100.0) (row "rktorch" "idle" 2 185.0)
                               (row "pytorch" "idle" 1 200.0) (row "pytorch" "idle" 2 400.0)
                               (row "rktorch" "train" 1 1.0) (row "rktorch" "train" 2 50.0)
                               (row "pytorch" "train" 1 2.0) (row "pytorch" "train" 2 100.0))))
    (check-regexp-match #rx"\\| idle \\| 1 \\| 100 \\| 1\\.00 \\| 200 \\| 1\\.00 \\| 0\\.50 \\| 11\\.0 \\| 11\\.0 \\|" md)
    (check-regexp-match #rx"\\| idle \\| 2 \\| 185 \\| 1\\.85 \\| 400 \\| 2\\.00 \\| 0\\.46 \\|" md)
    (check-regexp-match #rx"\\| train \\| 1 \\| 1 \\| 0\\.01 \\| 2 \\| 0\\.01 \\| 0\\.50 \\|" md)
    (check-regexp-match #rx"\\| train \\| 2 \\| 50 \\| 0\\.50 \\| 100 \\| 0\\.50 \\| 0\\.50 \\|" md))

  (define (e2e c impl variant s step #:gc [gc '()] #:settings [settings (hasheq)]
               #:reads [reads #f])
    (define r (hasheq 'suite "e2e" 'group "summary" 'case c 'impl impl 'variant variant
                      'unit "s/epoch" 'stats (hasheq 'median s) 'step_ms (list step)
                      'epoch_gc_ms gc 'settings settings 'steps_per_s (hasheq 'median 100.0)
                      'meta meta))
    (if reads (hash-set r 'reads reads) r))

  (test-case "C and D pair an example with its twins, variant by variant"
    (define records (list (e2e "17-workers" "rktorch" "crop-w4" 25.0 50.0)
                          (e2e "py-17-workers" "pytorch" "crop-w4" 5.0 10.0)
                          (e2e "05-mnist" "rktorch" #f 1.3 2.6)
                          (e2e "py-05-mnist" "pytorch" #f 1.0 2.0)
                          (e2e "py-05-mnist-per-param" "pytorch" #f 1.3 2.0)
                          (e2e "09-resnet" "rktorch" #f 14.0 30.0)))
    (define c (render 2 records))
    (check-regexp-match #rx"\\| 17-workers \\| crop-w4 \\| s/epoch \\| 25\\.00 \\| 5\\.00 \\| 5\\.00 \\| - \\| - \\| 50\\.00 \\| 10\\.00 \\| 5\\.00 \\| - \\| - \\| - \\|" c)
    (check-false (regexp-match? #rx"05-mnist" c))
    (define d (render 3 records))
    (check-regexp-match #rx"\\| 05-mnist \\| - \\| s/epoch \\| 1\\.30 \\| 1\\.00 \\| 1\\.30 \\| 1\\.30 \\| 1\\.00 \\| 2\\.60 \\| 2\\.00 \\| 1\\.30 \\| 2\\.00 \\| 1\\.30 \\| - \\|" d)
    (check-false (regexp-match? #rx"09-resnet|17-workers" d) "an example without a twin is left out")
    (check-false (render 3 (list (e2e "09-resnet" "rktorch" #f 14.0 30.0)))))

  (test-case "D sets no ratio when a setting reaches only one side of a pair"
    (define settings (hasheq 'EPOCHS "2" 'BASE "4" 'DATASET "fixture" 'RKTORCH_CIFAR10_DIR "/c"))
    (define d
      (render 3 (list (e2e "09-resnet" "rktorch" #f 14.0 30.0 #:settings settings
                           #:reads '("EPOCHS"))
                      (e2e "py-09-resnet" "pytorch" #f 7.0 15.0 #:settings settings
                           #:reads '("EPOCHS" "BATCH" "BASE" "DATASET" "OPTIMIZER")))))
    (check-regexp-match #rx"\\| 09-resnet \\| - \\| s/epoch \\| 14\\.00 \\| 7\\.00 \\| - \\|" d)
    (check-regexp-match #rx"\\| BASE DATASET \\|$" d))

  (test-case "E lists crossings, frees, tensors and op calls per phase"
    (check-regexp-match
     #rx"\\| 05-mnist \\| optimizer \\| 243 \\| 3 \\| 16 \\| 208 \\| batch=128 \\|"
     (render 4 (list (hasheq 'case "05-mnist" 'variant "optimizer" 'stats (hasheq 'median 243.0)
                             'free_stats (hasheq 'median 3.0) 'tensor_stats (hasheq 'median 16.0)
                             'op_call_stats (hasheq 'median 208.0) 'settings (hasheq 'batch 128)
                             'meta meta)))))

  (test-case "F sets the host proxy and the loss-item wait beside the synced step"
    (define (r group variant ms #:settings [settings (hasheq)] #:gc [gc #f])
      (define record (hasheq 'group group 'variant variant 'stats (hasheq 'median ms) 'meta meta
                             'settings settings 'samples (list ms (+ ms 1.0))))
      (if gc (hash-set record 'gc_ms gc) record))
    (define md
      (render 5 (list (r "phase" "item/forward" 4.0) (r "phase" "item/loss-item" 6.0)
                      (r "phase" "no-item/forward" 4.0) (r "phase" "phase-sync/forward" 9.0)
                      (r "step" "item" 10.0 #:gc '(2.0 3.0)) (r "step" "no-item" 8.0 #:gc '(0.0 0.0))
                      (r "step" "phase-sync" 12.0 #:gc '(1.0 1.0))
                      (r "step" "proxy" 7.5 #:settings (hasheq 'base 4) #:gc '(0.0 0.0)))))
    (check-regexp-match #rx"\\| forward \\| 4\\.00 \\| 4\\.00 \\| 9\\.00 \\|" md)
    (check-regexp-match #rx"\\| loss-item \\| 6\\.00 \\| - \\| - \\|" md)
    (check-regexp-match #rx"\\| step \\| 10\\.00 \\| 8\\.00 \\| 12\\.00 \\|" md)
    (check-regexp-match #rx"\\| host proxy: the step at base 4 \\| 7\\.50 \\| - \\| - \\|" md)
    (check-regexp-match #rx"\\| host proxy / step \\| 0\\.75 \\| - \\| - \\|" md)
    (check-regexp-match #rx"\\| GC in the step \\| 2\\.50 \\| 0\\.00 \\| 1\\.00 \\|" md)
    (check-regexp-match #rx"\\| step net of GC \\| 8\\.00 \\| 8\\.50 \\| 11\\.50 \\|" md)
    (check-regexp-match #rx"\\| GC in the host proxy \\| 0\\.00 \\| - \\| - \\|" md)
    (check-regexp-match #rx"\\| host proxy net of GC \\| 8\\.00 \\| - \\| - \\|" md)
    (check-regexp-match #rx"\\| host proxy / step, net of GC \\| 1\\.00 \\| - \\| - \\|" md)
    (check-regexp-match #rx"\\| device excess: the loss-item wait \\| 6\\.00 \\| - \\| - \\|" md))

  (test-case "G gives GC per epoch and its share of the epoch"
    (define md (render 6 (list (e2e "05-mnist" "rktorch" "no-troughs" 1.25 2.0 #:gc '(100 150)
                                    #:settings (hasheq 'RKTORCH_BENCH_COLLECT_AT_TROUGHS "0")))))
    (check-regexp-match
     #rx"\\| 05-mnist \\| no-troughs \\| RKTORCH_BENCH_COLLECT_AT_TROUGHS=0 \\| s/epoch \\| 1\\.25 \\| 100\\.00 \\| 125 \\| 10\\.0% \\|"
     md)
    (check-false (render 6 (list (e2e "05-mnist" "rktorch" #f 1.0 2.0))))))
