#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/string string-join)
         (only-in "markdown.rkt" num ref table whole))

(provide gc-row
         handoff-terms
         loss-row
         m1-markdown
         registration-rows
         valid-arms)

(define (median xs)
  (define v (sort (filter real? xs) <))
  (define n (length v))
  (cond
    [(zero? n) #f]
    [(odd? n) (list-ref v (quotient n 2))]
    [else (/ (+ (list-ref v (sub1 (quotient n 2))) (list-ref v (quotient n 2))) 2.0)]))

(define (block-key r) (list (ref r 'run_id) (ref r 'round) (ref r 'block)))

(define (clean? r limit)
  (define outside (ref r 'load_outside))
  (and (not (ref r 'error)) (not (and (real? outside) (real? limit) (>= outside limit)))))

(define (valid-arms records #:limit [limit #f])
  (define arms (filter (lambda (r) (equal? (ref r 'kind) "arm")) records))
  (define keys (remove-duplicates (map block-key arms)))
  (define blocks
    (for/list ([k (in-list keys)])
      (filter (lambda (r) (equal? (block-key r) k)) arms)))
  (define kept
    (filter (lambda (b) (andmap (lambda (r) (clean? r (or limit (ref r 'max_load)))) b)) blocks))
  (values (apply append kept) (- (length blocks) (length kept))))

(define (per-sample r . keys)
  (define ms (apply ref r keys))
  (and (real? ms) (/ (* 1000.0 ms) (ref r 'samples))))

(define (build-kind r) (if (zero? (ref r 'workers)) 'main_build 'worker_build))

(define (handoff-terms r)
  (define b (build-kind r))
  (define wall (per-sample r 'split b 'wall_ms))
  (define cpu (per-sample r 'split b 'cpu_ms))
  (define gc (per-sample r 'split b 'gc_ms))
  (hasheq 'us (ref r 'us_per_sample)
          'build_cpu cpu
          'build_gc gc
          'build_off (and wall cpu gc (- wall cpu gc))
          'collate (per-sample r 'split (if (zero? (ref r 'workers)) 'main_collate 'worker_collate)
                               'wall_ms)
          'step (per-sample r 'split 'step 'wall_ms)
          'next (per-sample r 'split 'next 'wall_ms)
          'next_cpu (per-sample r 'split 'next 'cpu_ms)
          'next_gc (per-sample r 'split 'next 'gc_ms)
          'start_finish (+ (per-sample r 'split 'start 'wall_ms) (per-sample r 'split 'finish 'wall_ms))
          'main_cpu (per-sample r 'main_cpu_ms)
          'head (per-sample r 'split 'worker_idle 'head_ms)
          'tail (per-sample r 'split 'worker_idle 'tail_ms)
          'gaps (per-sample r 'split 'worker_idle 'gaps_ms)
          'absent (per-sample r 'split 'worker_idle 'absent_ms)
          'gc_real (per-sample r 'gc 'all 'real_ms)
          'gc_cpu (per-sample r 'gc 'all 'cpu_ms)))

(define (medians rs f)
  (define hs (map f rs))
  (match hs
    ['() (hasheq)]
    [(cons h _)
     (for/hasheq ([k (in-hash-keys h)])
       (values k (median (map (lambda (h) (hash-ref h k)) hs))))]))

(define (cell rs pipeline workers impl #:trip [trip #f])
  (filter (lambda (r)
            (and (equal? (ref r 'pipeline) pipeline) (equal? (ref r 'workers) workers)
                 (equal? (ref r 'impl) impl) (equal? (ref r 'trip_mb) trip)))
          rs))

(define (us-of rs)
  (median (for/list ([r (in-list rs)])
            (or (ref r 'us_per_sample)
                (let ([rate (ref r 'samples_per_s)]) (and (real? rate) (/ 1e6 rate)))))))

(define (loss-row rs pipeline workers)
  (define m (medians (cell rs pipeline workers "rktorch") handoff-terms))
  (define m₀ (medians (cell rs pipeline 0 "rktorch") handoff-terms))
  (define py (us-of (cell rs pipeline workers "pytorch")))
  (define py₀ (us-of (cell rs pipeline 0 "pytorch")))
  (define (d k) (and (hash-ref m k #f) (hash-ref m₀ k #f) (/ (- (hash-ref m k) (hash-ref m₀ k)) workers)))
  (define (w k) (and (hash-ref m k #f) (/ (hash-ref m k) workers)))
  (cond
    [(or (zero? workers) (not (hash-ref m 'us #f)) (not (hash-ref m₀ 'us #f))) #f]
    [else
     (define loss (- (hash-ref m 'us) (/ (hash-ref m₀ 'us) workers)))
     (define serial (/ (- (hash-ref m₀ 'us) (hash-ref m₀ 'build_cpu) (hash-ref m₀ 'build_gc)
                          (hash-ref m₀ 'build_off))
                       workers))
     (define terms
       (list (d 'build_cpu) (d 'build_gc) (d 'build_off)
             (w 'head) (w 'tail) (w 'gaps) (w 'absent) (- serial)))
     (define explained (and (andmap real? terms) (apply + terms)))
     (hasheq 'pipeline pipeline 'workers workers
             'us (hash-ref m 'us) 'ideal (/ (hash-ref m₀ 'us) workers) 'loss loss
             'cpu (d 'build_cpu) 'gc (d 'build_gc) 'off (d 'build_off)
             'head (w 'head) 'tail (w 'tail) 'gaps (w 'gaps) 'absent (w 'absent) 'serial (- serial)
             'explained explained
             'py_us py 'py_loss (and py py₀ (- py (/ py₀ workers)))
             'next (hash-ref m 'next) 'next_cpu (hash-ref m 'next_cpu) 'next_gc (hash-ref m 'next_gc)
             'step (hash-ref m 'step) 'main_cpu (hash-ref m 'main_cpu)
             'gc_real (hash-ref m 'gc_real) 'gc_real₀ (hash-ref m₀ 'gc_real))]))

(define (mode-stat r mode key) (ref r 'gc mode key))

(define (slope points)
  (define xs (map car points))
  (define ys (map cdr points))
  (define n (length points))
  (cond
    [(< n 3) #f]
    [else
     (define mx (/ (apply + xs) n))
     (define my (/ (apply + ys) n))
     (define sxx (for/sum ([x (in-list xs)]) (* (- x mx) (- x mx))))
     (and (positive? sxx)
          (/ (for/sum ([x (in-list xs)] [y (in-list ys)]) (* (- x mx) (- y my))) sxx))]))

(define (gc-row rs pipeline workers trip)
  (define cs (cell rs pipeline workers "rktorch" #:trip trip))
  (define (med f) (median (map f cs)))
  (define (per-k r mode) (/ (* 1000.0 (mode-stat r mode 'count)) (ref r 'samples)))
  (define minor-points
    (for*/list ([r (in-list cs)] [p (in-list (ref r 'gc 'points))]
                #:when (match p [(list "minor" _ live) (real? live)] [_ #f]))
      (match-define (list _ ms live) p)
      (cons live ms)))
  (and (pair? cs)
       (hasheq 'pipeline pipeline 'workers workers 'trip trip 'blocks (length cs)
               'rate (med (lambda (r) (ref r 'samples_per_s)))
               'gc_us (med (lambda (r) (per-sample r 'gc 'all 'real_ms)))
               'gc_cpu_us (med (lambda (r) (per-sample r 'gc 'all 'cpu_ms)))
               'minor_per_k (med (lambda (r) (per-k r 'minor)))
               'major_per_k (med (lambda (r) (per-k r 'major)))
               'minor_ms (med (lambda (r) (mode-stat r 'minor 'mean_real_ms)))
               'major_ms (med (lambda (r) (mode-stat r 'major 'mean_real_ms)))
               'live (med (lambda (r) (mode-stat r 'all 'mean_live)))
               'us_per_k_live (let ([s (slope minor-points)]) (and s (* 1000.0 s))))))

(define registration-pieces
  '((guardian guardian raw) (lifetimes lifetimes raw) (registration registered raw)
    (accounting accounted registered) (rest facade accounted)))

(define (registration-rows records)
  (define ops (filter (lambda (r) (equal? (ref r 'kind) "op")) records))
  (define (at op layer n key)
    (median (for/list ([r (in-list ops)]
                       #:when (and (equal? (ref r 'op) op) (equal? (ref r 'layer) layer)
                                   (equal? (ref r 'threads) n)))
              (define per (ref r 'ops_per_call))
              (match key
                ['us (/ (ref r 'us_per_op) per)]
                ['kb (/ (or (ref r 'heap_bytes_per_op) 0) per 1024.0)]
                ['gc-free (/ (- (ref r 'us_per_op)
                                (/ (* 1000.0 (ref r 'gc_real_ms) (max 1 n)) (ref r 'ops)))
                             per)]))))
  (define threads (sort (remove-duplicates (map (lambda (r) (ref r 'threads)) ops)) <))
  (define layers
    (for/list ([l (in-list '("raw" "guardian" "lifetimes" "registered" "accounted" "facade"))])
      (cons l (for/list ([n (in-list threads)]) (list (at "mix" l n 'us) (at "mix" l n 'gc-free)
                                                      (at "mix" l n 'kb))))))
  (define (minus a b) (and (real? a) (real? b) (- a b)))
  (define pieces
    (for/list ([p (in-list registration-pieces)])
      (match-define (list name top base) p)
      (cons (symbol->string name)
            (for/list ([n (in-list threads)])
              (for/list ([key (in-list '(us gc-free kb))])
                (minus (at "mix" (symbol->string top) n key)
                       (at "mix" (symbol->string base) n key)))))))
  (values threads layers pieces
          (for/list ([n (in-list threads)]) (at "cifar-chain" "facade" n 'us))))

(define (triple-text t)
  (match-define (list us free kb) t)
  (format "~a / ~a / ~a" (num us 2) (num free 2) (num kb 2)))

(define (registration-markdown records)
  (define-values (threads layers pieces chain) (registration-rows records))
  (define by-threads (map (lambda (n) (format "~a threads" n)) threads))
  (define header (cons "µs/op, excl. GC / KB/op" by-threads))
  (define (rows xs)
    (for/list ([x (in-list xs)])
      (match-define (cons name triples) x)
      (cons name (map triple-text triples))))
  (string-append
   "### Registration and accounting per op, the loader's op mix\n\n"
   (table header (rows layers)) "\n\n"
   (table (cons "piece" by-threads) (rows pieces)) "\n\n"
   (table (cons "cifar chain, µs/sample" by-threads) (list (cons "facade" (map num chain))))
   "\n"))

(define (loss-markdown rs pipelines workers)
  (define header
    '("pipeline" "w" "µs/sample" "ideal" "loss" "build CPU Δ" "build GC Δ" "build off-CPU Δ"
      "idle head" "idle tail" "idle gaps" "absent" "serial w0" "sum" "py loss"
      "main next (CPU, GC)" "main step"))
  (define rows
    (for*/list ([p (in-list pipelines)] [w (in-list workers)]
                #:when (positive? w)
                [row (in-value (loss-row rs p w))] #:when row)
      (define (n k) (whole (hash-ref row k)))
      (list p (number->string w) (n 'us) (n 'ideal) (n 'loss) (n 'cpu) (n 'gc) (n 'off)
            (n 'head) (n 'tail) (n 'gaps) (n 'absent) (n 'serial) (n 'explained) (n 'py_loss)
            (format "~a (~a, ~a)" (n 'next) (n 'next_cpu) (n 'next_gc)) (n 'step))))
  (string-append "### Hand-off loss per sample, µs (Δ: worker-summed against w0, ÷ w)\n\n"
                 (table header rows) "\n"))

(define (gc-markdown rs pipelines workers trips)
  (define header
    '("pipeline" "w" "trip MB" "arms" "samples/s" "GC µs/sample" "GC CPU µs/sample"
      "minor /1k" "major /1k" "ms/minor" "ms/major" "live entries" "ms/minor per 1k live"))
  (define rows
    (for*/list ([p (in-list pipelines)] [w (in-list workers)] [t (in-list trips)]
                [row (in-value (gc-row rs p w t))] #:when row)
      (define (n k [d 1]) (num (hash-ref row k) d))
      (list p (number->string w) (if t (number->string t) "default") (number->string (hash-ref row 'blocks))
            (whole (hash-ref row 'rate)) (n 'gc_us) (n 'gc_cpu_us) (n 'minor_per_k) (n 'major_per_k 2) (n 'minor_ms 2)
            (n 'major_ms) (whole (hash-ref row 'live)) (n 'us_per_k_live))))
  (string-append "### GC against live objects\n\n" (table header rows) "\n"))

(define (m1-markdown records #:limit [limit #f])
  (define-values (rs dropped) (valid-arms records #:limit limit))
  (define groups
    (remove-duplicates (for/list ([r (in-list rs)]) (list (ref r 'tree) (ref r 'racket)))))
  (define (values-of key) (remove-duplicates (filter values (map (lambda (r) (ref r key)) rs))))
  (define pipelines (sort (values-of 'pipeline) string<?))
  (define workers (sort (values-of 'workers) <))
  (define trips (cons #f (sort (values-of 'trip_mb) <)))
  (string-join
   (append
    (list (format "Blocks dropped for outside load (~a busy cores) or a failed arm: ~a\n"
                  (or limit "each run's limit") dropped))
    (for/list ([g (in-list groups)])
      (match-define (list tree racket) g)
      (define in-g (filter (lambda (r) (and (equal? (ref r 'tree) tree) (equal? (ref r 'racket) racket))) rs))
      (format "## ~a, Racket ~a\n\n~a\n~a" tree racket
              (loss-markdown in-g pipelines workers) (gc-markdown in-g pipelines workers trips)))
    (if (ormap (lambda (r) (equal? (ref r 'kind) "op")) records)
        (list (registration-markdown records))
        '()))
   "\n"))
