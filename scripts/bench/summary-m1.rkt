#lang racket/base

(require (only-in racket/list append* remove-duplicates)
         (only-in racket/match match match* match-define)
         (only-in racket/string string-join)
         (only-in "harness.rkt" list-median)
         (only-in "markdown.rkt" column-table num ref table whole)
         (only-in "summary-blocks.rkt" arm-blocks complete-blocks outside-clean?)
         (only-in "summary-m0.rkt" block-arms))

(provide block-loss
         gc-row
         handoff-terms
         loss-row
         m1-arms
         m1-markdown
         registration-pieces
         registration-rows
         valid-arms)

(define (m1-arms pipeline workers trip)
  (if trip '(rktorch) (block-arms pipeline workers)))

(define (expected-arms b)
  (match-define (cons r _) (hash-values b))
  (map symbol->string
       (m1-arms (string->symbol (ref r 'pipeline)) (ref r 'workers)
                (ref (hash-ref b "rktorch" #f) 'trip_mb))))

(define (valid-arms records #:limit [limit #f])
  (define arms (filter (lambda (r) (equal? (ref r 'kind) "arm")) records))
  (define-values (kept dropped)
    (complete-blocks (arm-blocks arms)
                     #:expected expected-arms
                     #:clean? (lambda (r) (outside-clean? r (or limit (ref r 'max_load))))))
  (values (append* (map hash-values kept)) dropped))

(define (per-sample r . keys)
  (define ms (apply ref r keys))
  (and (real? ms) (/ (* 1000.0 ms) (ref r 'samples))))

(define (us-of r)
  (or (ref r 'us_per_sample)
      (let ([rate (ref r 'samples_per_s)]) (and (real? rate) (positive? rate) (/ 1e6 rate)))))

(define (handoff-terms r)
  (define main? (zero? (ref r 'workers)))
  (define build (if main? 'main_build 'worker_build))
  (hasheq 'us (us-of r)
          'build_wall (per-sample r 'split build 'wall_ms)
          'build_cpu (per-sample r 'split build 'cpu_ms)
          'build_off (per-sample r 'split build 'off_ms)
          'build_gc (per-sample r 'split build 'gc_ms)
          'outside (or (per-sample r 'split build 'outside_ms) 0.0)
          'collate (per-sample r 'split (if main? 'main_collate 'worker_collate) 'wall_ms)
          'step (per-sample r 'split 'step 'wall_ms)
          'next (per-sample r 'split 'next 'wall_ms)
          'next_cpu (per-sample r 'split 'next 'cpu_ms)
          'next_gc (per-sample r 'split 'next 'gc_ms)
          'main_cpu (per-sample r 'main_cpu_ms)
          'head (per-sample r 'split 'worker_idle 'head_ms)
          'tail (per-sample r 'split 'worker_idle 'tail_ms)
          'gaps (per-sample r 'split 'worker_idle 'gaps_ms)
          'absent (per-sample r 'split 'worker_idle 'absent_ms)
          'gc_real (per-sample r 'gc 'all 'real_ms)
          'gc_cpu (per-sample r 'gc 'all 'cpu_ms)))

(define summed-terms '(cpu off outside head tail gaps absent serial))

(define (block-loss t t₀ workers)
  (define (at h k) (hash-ref h k #f))
  (define needed '(us build_wall build_cpu build_off build_gc outside head tail gaps absent))
  (cond
    [(not (and (andmap (lambda (k) (real? (at t k))) needed)
               (andmap (lambda (k) (real? (at t₀ k))) '(us build_wall build_cpu build_off build_gc))))
     #f]
    [else
     (define (d k) (/ (- (at t k) (at t₀ k)) workers))
     (define (w k) (/ (at t k) workers))
     (define terms
       (hasheq 'cpu (d 'build_cpu)
               'off (d 'build_off)
               'outside (- (w 'outside))
               'head (w 'head) 'tail (w 'tail) 'gaps (w 'gaps) 'absent (w 'absent)
               'serial (- (/ (- (at t₀ 'us) (at t₀ 'build_wall)) workers))))
     (define loss (- (at t 'us) (/ (at t₀ 'us) workers)))
     (hash-set* terms
                'us (at t 'us)
                'ideal (/ (at t₀ 'us) workers)
                'loss loss
                'gc (d 'build_gc)
                'residual (- loss (for/sum ([k (in-list summed-terms)]) (hash-ref terms k))))]))

(define (median-by hs)
  (match hs
    ['() (hasheq)]
    [(cons h _)
     (for/hasheq ([k (in-hash-keys h)])
       (values k (list-median (map (lambda (h) (hash-ref h k #f)) hs))))]))

(define (cell rs pipeline workers arm #:trip [trip #f])
  (filter (lambda (r)
            (and (equal? (ref r 'pipeline) pipeline) (equal? (ref r 'workers) workers)
                 (equal? (ref r 'arm) arm) (equal? (ref r 'trip_mb) trip)))
          rs))

(define (arm-us rs pipeline workers arm)
  (list-median (map us-of (cell rs pipeline workers arm))))

(define (w0-reference rs pipeline)
  (define m (median-by (map handoff-terms (cell rs pipeline 0 "rktorch"))))
  (match m
    [(hash-table ('build_wall (? real? wall)) ('build_cpu (? real? cpu)) (_ _) ...)
     (hash-set m 'build_off (- wall cpu))]
    [_ m]))

(define (loss-row rs pipeline workers)
  (define t₀ (w0-reference rs pipeline))
  (define ts (map handoff-terms (cell rs pipeline workers "rktorch")))
  (define blocks (filter values (map (lambda (t) (block-loss t t₀ workers)) ts)))
  (define py₀ (arm-us rs pipeline 0 "pytorch"))
  (define (py-loss arm)
    (define py (arm-us rs pipeline workers arm))
    (and py py₀ (- py (/ py₀ workers))))
  (define m (median-by ts))
  (and (positive? workers) (pair? blocks)
       (hash-set* (median-by blocks)
                  'pipeline pipeline 'workers workers 'blocks (length blocks)
                  'py_loss (py-loss "pytorch")
                  'py_persistent_loss (py-loss "pytorch-persistent")
                  'next (hash-ref m 'next) 'next_cpu (hash-ref m 'next_cpu)
                  'next_gc (hash-ref m 'next_gc) 'step (hash-ref m 'step))))

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
  (define (med f) (list-median (map f cs)))
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
  '((free raw kept) (guardian guardian kept) (lifetimes lifetimes raw)
    (registration registered kept) (accounting accounted registered) (rest facade accounted)))

(define registration-layers '("kept" "raw" "guardian" "lifetimes" "registered" "accounted" "facade"))

(define (registration-rows records)
  (define ops (filter (lambda (r) (equal? (ref r 'kind) "op")) records))
  (define (at op layer n key)
    (list-median
     (for/list ([r (in-list ops)]
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
    (for/list ([l (in-list registration-layers)])
      (cons l (for/list ([n (in-list threads)])
                (for/list ([key (in-list '(us gc-free kb))]) (at "mix" l n key))))))
  (define (minus a b) (and (real? a) (real? b) (- a b)))
  (define pieces
    (for/list ([p (in-list registration-pieces)])
      (match-define (list name top base) p)
      (cons (symbol->string name)
            (for/list ([n (in-list threads)])
              (for/list ([key (in-list '(us gc-free kb))])
                (minus (at "mix" (symbol->string top) n key)
                       (at "mix" (symbol->string base) n key)))))))
  (define (backlog n)
    (define kept
      (filter (lambda (r)
                (and (equal? (ref r 'op) "mix") (equal? (ref r 'layer) "kept")
                     (equal? (ref r 'threads) n) (real? (ref r 'kept_pending_peak))))
              ops))
    (match kept
      ['() #f]
      [_
       (define peak (apply max (map (lambda (r) (ref r 'kept_pending_peak)) kept)))
       (define limit (* (max 1 n) (apply min (map (lambda (r) (ref r 'kept_batch)) kept))))
       (cons peak (> peak limit))]))
  (values threads layers pieces
          (for/list ([n (in-list threads)]) (at "cifar-chain" "facade" n 'us))
          (map backlog threads)))

(define (triple-text t)
  (match-define (list us free kb) t)
  (format "~a / ~a / ~a" (num us 2) (num free 2) (num kb 2)))

(define (registration-markdown records racket)
  (define-values (threads layers pieces chain backlog) (registration-rows records))
  (define by-threads (map (lambda (n) (format "~a threads" n)) threads))
  (define (rows xs)
    (for/list ([x (in-list xs)])
      (match-define (cons name triples) x)
      (cons name
            (for/list ([t (in-list triples)] [b (in-list backlog)])
              (match* (name b)
                [("free" (cons _ #t)) (string-append (triple-text t) " (backlog)")]
                [(_ _) (triple-text t)])))))
  (define peaks
    (cons "kept backlog peak, handles"
          (map (lambda (b) (num (and b (car b)) 0)) backlog)))
  (define on (if racket (string-append ", Racket " racket) ""))
  (string-append
   "### Registration and accounting per op, the loader's op mix" on "\n\n"
   (table (cons "µs/op, excl. GC / KB/op" by-threads) (append (rows layers) (list peaks))) "\n\n"
   (table (cons "piece" by-threads) (rows pieces)) "\n\n"
   (if (ormap (lambda (b) (and b (cdr b))) backlog)
       (string-append "(backlog): kept's freeing thread fell more than a batch per thread behind, "
                      "so free is biased low there.\n\n")
       "")
   (table (cons "cifar chain, µs/sample" by-threads) (list (cons "facade" (map num chain))))
   "\n"))

(define (registration-sections records)
  (define rackets
    (for/list ([r (in-list records)] #:when (equal? (ref r 'kind) "op")) (ref r 'racket)))
  (for/list ([v (in-list (remove-duplicates rackets))])
    (registration-markdown (filter (lambda (r) (equal? (ref r 'racket) v)) records) v)))

(define (column header format-of key)
  (cons header (lambda (row) (format-of (hash-ref row key)))))

(define (us header key) (column header whole key))

(define loss-columns
  (list (column "pipeline" values 'pipeline)
        (column "w" number->string 'workers)
        (column "blocks" number->string 'blocks)
        (us "µs/sample" 'us) (us "ideal" 'ideal) (us "loss" 'loss)
        (us "build CPU Δ" 'cpu) (us "build off-CPU Δ" 'off) (us "builds outside epochs" 'outside)
        (us "idle head" 'head) (us "idle tail" 'tail) (us "idle gaps" 'gaps)
        (us "absent" 'absent) (us "serial w0" 'serial) (us "residual" 'residual)
        (us "build GC overlap Δ" 'gc)
        (us "py loss" 'py_loss) (us "py persistent loss" 'py_persistent_loss)
        (cons "main next (CPU, GC)"
              (lambda (row)
                (format "~a (~a, ~a)" (whole (hash-ref row 'next)) (whole (hash-ref row 'next_cpu))
                        (whole (hash-ref row 'next_gc)))))
        (us "main step" 'step)))

(define (loss-markdown rs pipelines workers)
  (define rows
    (for*/list ([p (in-list pipelines)] [w (in-list workers)]
                #:when (positive? w)
                [row (in-value (loss-row rs p w))] #:when row)
      row))
  (string-append "### Hand-off loss per sample, µs (Δ: worker-summed against w0, ÷ w; "
                 "medians over blocks)\n\n"
                 (column-table loss-columns rows) "\n"))

(define (fixed header key [digits 1]) (column header (lambda (x) (num x digits)) key))

(define gc-columns
  (list (column "pipeline" values 'pipeline)
        (column "w" number->string 'workers)
        (column "trip MB" (lambda (t) (if t (number->string t) "default")) 'trip)
        (column "arms" number->string 'blocks)
        (column "samples/s" whole 'rate)
        (fixed "GC µs/sample" 'gc_us) (fixed "GC CPU µs/sample" 'gc_cpu_us)
        (fixed "minor /1k" 'minor_per_k) (fixed "major /1k" 'major_per_k 2)
        (fixed "ms/minor" 'minor_ms 2) (fixed "ms/major" 'major_ms)
        (column "live entries" whole 'live)
        (fixed "ms/minor per 1k live" 'us_per_k_live)))

(define (gc-markdown rs pipelines workers trips)
  (define rows
    (for*/list ([p (in-list pipelines)] [w (in-list workers)] [t (in-list trips)]
                [row (in-value (gc-row rs p w t))] #:when row)
      row))
  (string-append "### GC against live objects\n\n" (column-table gc-columns rows) "\n"))

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
    (list (format "Blocks dropped for outside load (~a busy cores), a failed or a missing arm: ~a\n"
                  (or limit "each run's limit") dropped))
    (for/list ([g (in-list groups)])
      (match-define (list tree racket) g)
      (define in-g (filter (lambda (r) (and (equal? (ref r 'tree) tree) (equal? (ref r 'racket) racket))) rs))
      (format "## ~a, Racket ~a\n\n~a\n~a" tree racket
              (loss-markdown in-g pipelines workers) (gc-markdown in-g pipelines workers trips)))
    (registration-sections records))
   "\n"))
