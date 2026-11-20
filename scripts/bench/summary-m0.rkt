#lang racket/base

(require (only-in racket/list argmax filter-map remove-duplicates)
         (only-in racket/string string-join)
         (only-in threading ~>)
         (only-in "harness.rkt" list-median)
         (only-in "markdown.rkt" host num ref table whole)
         (only-in "summary-op-cost.rkt" k3-thread-scaling k6-gc-per-sample))

(provide arm-rate
         block-ratios
         cell-row
         done-when
         m0-markdown
         m0-records
         scaling
         training-row
         valid-blocks)

(define table-h '("cifar-png" "cifar-batched" "jpeg"))
(define like-for-like "pytorch")
(define persistent "pytorch-persistent")
(define scaling-bar 3.0)
(define ratio-bar 0.5)
(define k3-bar 2.0)

(define (kind records k)
  (filter (lambda (r) (equal? (ref r 'kind) k)) records))

(define (m0-records records)
  (kind records "arm"))

(define (rate-key pipeline)
  (if (equal? pipeline "train") 'main_steps_per_s 'samples_per_s))

(define (versions records)
  (~> records m0-records (map (lambda (r) (ref r 'racket)) _) remove-duplicates))

(define (of-version records v)
  (filter (lambda (r) (equal? (ref r 'racket) v)) records))

(define (block-key r)
  (list (ref r 'run_id) (ref r 'round) (ref r 'block)))

(define (cell-blocks records pipeline workers)
  (define arms
    (filter (lambda (r) (and (equal? (ref r 'pipeline) pipeline) (equal? (ref r 'workers) workers)))
            (m0-records records)))
  (for/list ([key (in-list (remove-duplicates (map block-key arms)))])
    (for/hash ([r (in-list arms)] #:when (equal? (block-key r) key))
      (values (ref r 'arm) r))))

(define (clean-arm? r)
  (define outside (ref r 'load_outside))
  (define limit (ref r 'max_load))
  (and (not (ref r 'error))
       (real? (ref r (rate-key (ref r 'pipeline))))
       (not (and (real? outside) (real? limit) (>= outside limit)))))

(define (valid-blocks records pipeline workers)
  (define blocks (cell-blocks records pipeline workers))
  (define valid
    (filter (lambda (b) (and (hash-ref b "rktorch" #f) (andmap clean-arm? (hash-values b))))
            blocks))
  (values valid (- (length blocks) (length valid))))

(define (block-ratios records pipeline workers arm)
  (define-values (valid _dropped) (valid-blocks records pipeline workers))
  (define key (rate-key pipeline))
  (for*/list ([b (in-list valid)]
              [other (in-value (hash-ref b arm #f))]
              #:when (and other (positive? (ref other key))))
    (/ (ref (hash-ref b "rktorch") key) (ref other key))))

(define (arm-values records pipeline workers arm key)
  (define-values (valid _dropped) (valid-blocks records pipeline workers))
  (filter-map (lambda (b) (ref (hash-ref b arm #f) key)) valid))

(define (arm-rate records pipeline workers arm)
  (list-median (arm-values records pipeline workers arm (rate-key pipeline))))

(define (per-worker r workers)
  (define active (ref r 'os_threads_active))
  (define idle (ref r 'os_threads_idle))
  (and (positive? workers) (real? active) (real? idle) (/ (- active idle) workers 1.0)))

(define (worker-threads records pipeline workers arm)
  (define-values (valid _dropped) (valid-blocks records pipeline workers))
  (define counts
    (for*/list ([b (in-list valid)]
                [r (in-value (hash-ref b arm #f))]
                #:when r)
      (cond
        [(equal? arm "rktorch") (per-worker r workers)]
        [else
         (define ws (filter real? (or (ref r 'worker_threads) '())))
         (and (pair? ws) (apply max ws))])))
  (list-median counts))

(define (cell-row records pipeline workers)
  (define-values (valid dropped) (valid-blocks records pipeline workers))
  (define (m arm key) (list-median (arm-values records pipeline workers arm key)))
  (hasheq 'pipeline pipeline 'workers workers 'blocks (length valid) 'dropped dropped
          'rktorch (arm-rate records pipeline workers "rktorch")
          'pytorch (arm-rate records pipeline workers like-for-like)
          'persistent (arm-rate records pipeline workers persistent)
          'vs-pytorch (list-median (block-ratios records pipeline workers like-for-like))
          'vs-persistent (list-median (block-ratios records pipeline workers persistent))
          'rk-gc-ms (m "rktorch" 'gc_ms) 'rk-gc-count (m "rktorch" 'gc_count)
          'py-gc-ms (m like-for-like 'gc_ms) 'py-gc-count (m like-for-like 'gc_count)
          'rk-threads (worker-threads records pipeline workers "rktorch")
          'py-threads (worker-threads records pipeline workers like-for-like)))

(define (scaling records pipeline arm #:from [lo 0] #:to [hi 4])
  (define top (arm-rate records pipeline hi arm))
  (define base (arm-rate records pipeline lo (if (equal? arm persistent) like-for-like arm)))
  (and (real? top) (real? base) (positive? base) (/ top base)))

(define (training-row records workers)
  (define (ops arm) (list-median (arm-values records "train" workers arm 'ops_per_s)))
  (define-values (valid dropped) (valid-blocks records "train" workers))
  (hasheq 'workers workers 'blocks (length valid) 'dropped dropped
          'rktorch (arm-rate records "train" workers "rktorch")
          'pytorch (arm-rate records "train" workers like-for-like)
          'vs-pytorch (list-median (block-ratios records "train" workers like-for-like))
          'rk-ops (ops "rktorch") 'py-ops (ops like-for-like)))

(define (verdict value ok?)
  (cond
    [(not (real? value)) "n/a"]
    [(ok? value) "met"]
    [else "unmet"]))

(define (k3-worst op-cost)
  (define rows (filter (lambda (r) (real? (hash-ref r 'value))) (k3-thread-scaling op-cost)))
  (and (pair? rows) (argmax (lambda (r) (hash-ref r 'value)) rows)))

(define (done-when records op-cost)
  (define (row target cell value ok?)
    (list target cell value (verdict value ok?)))
  (define worst (k3-worst op-cost))
  (append
   (for/list ([p (in-list table-h)])
     (row "Table H w4 ≥ 3× w0" p (scaling records p "rktorch") (lambda (x) (>= x scaling-bar))))
   (for/list ([p (in-list table-h)])
     (row "rk ÷ py ≥ 0.5 at w4 (non-persistent twin)" p
          (list-median (block-ratios records p 4 like-for-like))
          (lambda (x) (>= x ratio-bar))))
   (list (row "main's training rate at w4 ≥ PyTorch's" "train"
              (list-median (block-ratios records "train" 4 like-for-like))
              (lambda (x) (>= x 1.0)))
         (row "K3 ≤ 2×" (if worst (format "~a, worst op" (hash-ref worst 'op)) "-")
              (and worst (hash-ref worst 'value))
              (lambda (x) (<= x k3-bar))))))

(define (cells-of records pipeline)
  (~> (m0-records records)
      (filter (lambda (r) (equal? (ref r 'pipeline) pipeline)) _)
      (map (lambda (r) (ref r 'workers)) _)
      remove-duplicates
      (sort <)))

(define (pipelines-of records)
  (~> (m0-records records)
      (map (lambda (r) (ref r 'pipeline)) _)
      remove-duplicates
      (filter (lambda (p) (not (equal? p "train"))) _)))

(define (blocks-text row)
  (define dropped (hash-ref row 'dropped))
  (if (positive? dropped)
      (format "~a (~a dropped)" (hash-ref row 'blocks) dropped)
      (number->string (hash-ref row 'blocks))))

(define (k6-column records pipeline)
  (define rk (for/list ([r (in-list (m0-records records))]
                        #:when (and (equal? (ref r 'arm) "rktorch") (clean-arm? r)))
               (hash-set r 'kind "pipeline")))
  (for/hash ([row (in-list (k6-gc-per-sample rk #:pipeline pipeline))])
    (values (hash-ref row 'workers) (hash-ref row 'value))))

(define (h-table records)
  (table (list "pipeline" "workers" "rk/s" "py/s" "py persistent/s" "rk ÷ py" "rk ÷ py persistent"
               "blocks" "rk GC ms" "rk GCs" "rk GC µs/sample" "py GC ms" "py GCs"
               "rk threads/worker" "py threads/worker")
         (for*/list ([p (in-list (pipelines-of records))]
                     [k6 (in-value (k6-column records p))]
                     [w (in-list (cells-of records p))])
           (define c (cell-row records p w))
           (define (at key [digits 2]) (num (hash-ref c key) digits))
           (list p (number->string w) (whole (hash-ref c 'rktorch)) (whole (hash-ref c 'pytorch))
                 (whole (hash-ref c 'persistent)) (at 'vs-pytorch) (at 'vs-persistent)
                 (blocks-text c) (whole (hash-ref c 'rk-gc-ms)) (whole (hash-ref c 'rk-gc-count))
                 (num (hash-ref k6 w #f) 1) (whole (hash-ref c 'py-gc-ms))
                 (whole (hash-ref c 'py-gc-count)) (at 'rk-threads 1) (at 'py-threads 1)))))

(define (scaling-table records)
  (define tops (for/list ([p (in-list (pipelines-of records))]) (apply max 0 (cells-of records p))))
  (table (list "pipeline" "rk w4 ÷ w0" "py w4 ÷ w0" "py persistent w4 ÷ py w0" "rk wmax ÷ w0"
               "py wmax ÷ w0")
         (for/list ([p (in-list (pipelines-of records))] [top (in-list tops)])
           (define (s arm #:to [hi 4]) (num (scaling records p arm #:to hi) 2))
           (list p (s "rktorch") (s like-for-like) (s persistent)
                 (format "~a (w~a)" (s "rktorch" #:to top) top) (s like-for-like #:to top)))))

(define (training-table records)
  (table (list "workers" "rk main steps/s" "py main steps/s" "rk ÷ py" "rk worker ops/s"
               "py worker ops/s" "blocks")
         (for/list ([w (in-list (cells-of records "train"))])
           (define t (training-row records w))
           (list (number->string w) (whole (hash-ref t 'rktorch)) (whole (hash-ref t 'pytorch))
                 (num (hash-ref t 'vs-pytorch) 2) (whole (hash-ref t 'rk-ops))
                 (whole (hash-ref t 'py-ops)) (blocks-text t)))))

(define (k3-table op-cost)
  (table (list "op" "1 thread µs/op" "8 threads µs/op" "K3")
         (for/list ([r (in-list (k3-thread-scaling op-cost))])
           (list (hash-ref r 'op) (num (hash-ref r 'one) 2) (num (hash-ref r 'many) 2)
                 (num (hash-ref r 'value) 2)))))

(define (heading records)
  (define meta (for/first ([r (in-list records)] #:when (equal? (ref r 'kind) "meta")) r))
  (define arms (m0-records records))
  (define loads (filter real? (map (lambda (r) (ref r 'load_outside)) arms)))
  (format "~a; runs ~a; rounds ~a; max outside load ~a, the most seen ~a"
          (if meta (host meta) "?")
          (string-join (map (lambda (x) (format "~a" x))
                            (remove-duplicates (map (lambda (r) (ref r 'run_id)) arms)))
                       ", ")
          (add1 (apply max -1 (filter real? (map (lambda (r) (ref r 'round)) arms))))
          (or (for/first ([r (in-list arms)] #:when (ref r 'max_load)) (ref r 'max_load)) "-")
          (if (pair? loads) (num (apply max loads) 1) "-")))

(define (version-section records op-cost v)
  (define rs (of-version records v))
  (define ks (of-version op-cost v))
  (string-join
   (list (format "## Racket ~a" v)
         (heading rs)
         "### Done when"
         (table (list "target" "cell" "value" "verdict")
                (for/list ([row (in-list (done-when rs ks))])
                  (list (car row) (cadr row) (num (caddr row) 2) (cadddr row))))
         "### Table H: loaders against DataLoader, samples/s"
         (h-table rs)
         "### Scaling"
         (scaling-table rs)
         "### #439: main's training rate while workers run"
         (training-table rs)
         "### K3: per-op cost at 8 threads ÷ 1 thread (op-cost.rkt)"
         (if (pair? ks) (k3-table ks) "no op-cost records"))
   "\n\n"))

(define (version-split records)
  (define vs (versions records))
  (table (append (list "pipeline" "workers") (map (lambda (v) (format "rk ÷ py, ~a" v)) vs))
         (for*/list ([p (in-list (append (pipelines-of records) '("train")))]
                     [w (in-list (cells-of records p))])
           (append (list p (number->string w))
                   (for/list ([v (in-list vs)])
                     (~> (of-version records v)
                         (block-ratios p w like-for-like)
                         list-median
                         (num 2)))))))

(define (m0-markdown records #:op-cost [op-cost '()])
  (string-join
   (append (list "# M0: the gap to PyTorch (#555)"
                 (string-append
                  "Each block runs one cell's arms back to back in shuffled order; a ratio is "
                  "the median over blocks of rktorch over that block's PyTorch arm. `py` is "
                  "the twin with `persistent_workers=False`, like for like with rktorch's "
                  "loader (#536); `py persistent` keeps its workers across epochs. A block "
                  "with an error, or with an arm whose outside load reached the limit, is "
                  "dropped. PyTorch's GC counts its main process only."))
           (map (lambda (v) (version-section records op-cost v)) (versions records))
           (if (~> (versions records) length (> 1))
               (list "## rk ÷ py by Racket version" (version-split records))
               '()))
   "\n\n"))
