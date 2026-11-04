#lang racket/base

(require (only-in racket/list filter-map last remove-duplicates)
         (only-in racket/string string-join)
         (only-in threading ~>)
         (only-in "harness.rkt" list-median)
         (only-in "markdown.rkt" commit host num ref table whole)
         (only-in "op-cost-layers.rkt"
                  ladder layer-pieces leave-one-out loo-base pure-overhead-ops))

(provide samples-by-rep
         layer-median
         paired-delta
         piece-label
         attribution
         crossing-us
         k1-overhead
         k2-pytorch-ratio
         k3-thread-scaling
         k4-heap
         k5-pipelines
         k6-gc-share
         gap-tolerance
         op-cost-markdown)

(define gap-tolerance 0.10)

(define (kind records k)
  (filter (lambda (r) (equal? (ref r 'kind) k)) records))

(define (op-rows records #:impl impl #:op op #:layer layer #:threads threads)
  (filter (lambda (r) (and (equal? (ref r 'impl) impl) (equal? (ref r 'op) op)
                           (equal? (ref r 'layer) layer) (equal? (ref r 'threads) threads)))
          (kind records "op")))

(define (samples-by-rep rows key)
  (define reps (remove-duplicates (map (lambda (r) (ref r 'rep)) rows)))
  (for*/hash ([rep (in-list reps)]
              [m (in-value (list-median (for/list ([r (in-list rows)]
                                                    #:when (equal? (ref r 'rep) rep))
                                          (ref r key))))]
              #:when m)
    (values rep m)))

(define (layer-median records op layer
                      #:impl [impl "rktorch"] #:threads [threads 0] #:key [key 'us_per_op])
  (~> (op-rows records #:impl impl #:op op #:layer layer #:threads threads)
      (samples-by-rep key)
      hash-values
      list-median))

(define (paired-delta upper lower)
  (list-median (for/list ([(rep u) (in-hash upper)] #:when (hash-ref lower rep #f))
                 (- u (hash-ref lower rep)))))

(define (piece-label layers)
  (string-join (map (lambda (l) (hash-ref layer-pieces l (symbol->string l))) layers) " + "))

(define (by-rep records op layer key)
  (samples-by-rep (op-rows records #:impl "rktorch" #:op op #:layer (symbol->string layer)
                           #:threads 0)
                  key))

(define (present-layers records op)
  (filter (lambda (l) (~> (by-rep records op l 'us_per_op) hash-count positive?)) ladder))

(define (counts records op layer key)
  (define row (findf (lambda (r) (and (equal? (ref r 'op) op) (equal? (ref r 'layer) layer)))
                     (kind records "counts")))
  (and row (ref row key)))

(define (step records op prev cur key)
  (cond
    [prev (paired-delta (by-rep records op cur key) (by-rep records op prev key))]
    [else (layer-median records op (symbol->string cur) #:key key)]))

(define (crossing-step records op prev cur)
  (define c (counts records op (symbol->string cur) 'crossings_per_op))
  (define p (and prev (counts records op (symbol->string prev) 'crossings_per_op)))
  (cond
    [(not c) #f]
    [(not prev) c]
    [p (- c p)]
    [else #f]))

(define (spanned-layers prev cur)
  (for/list ([l (in-list (member cur (reverse ladder)))]
             #:break (eq? l prev))
    l))

;; Taking one piece out of the whole stack prices it independently of the
;; ladder, which prices it on top of the layers below only.
(define (loo-check records op piece base-total)
  (define layer (string->symbol (ref piece 'layer)))
  (define without (and (= 1 (length (ref piece 'spanned))) (hash-ref leave-one-out layer #f)))
  (define loo (and without
                   (paired-delta (by-rep records op loo-base 'us_per_op)
                                 (by-rep records op without 'us_per_op))))
  (define ladder-us (ref piece 'us))
  (and loo
       (hasheq 'layer (ref piece 'layer)
               'piece (ref piece 'piece)
               'ladder ladder-us
               'loo loo
               'agrees (and (real? ladder-us) (real? base-total) (positive? base-total)
                            (<= (abs (- ladder-us loo)) (* gap-tolerance base-total))))))

(define (contract-cost records op)
  (define dispatch (by-rep records op 'unchecked 'us_per_op))
  (and (positive? (hash-count dispatch))
       (paired-delta (by-rep records op 'contracted 'us_per_op) dispatch)))

(define (attribution records op)
  (define layers (present-layers records op))
  (define pieces
    (for/list ([cur (in-list layers)]
               [prev (in-list (cons #f layers))])
      (define spanned (reverse (spanned-layers prev cur)))
      (hasheq 'piece (piece-label spanned)
              'layer (symbol->string cur)
              'spanned spanned
              'us (step records op prev cur 'us_per_op)
              'heap (step records op prev cur 'heap_bytes_per_op)
              'crossings (crossing-step records op prev cur))))
  (define top (and (pair? layers) (last layers)))
  (define total (and top (layer-median records op (symbol->string top))))
  (define sum (and (pair? pieces) (andmap (lambda (p) (real? (ref p 'us))) pieces)
                   (for/sum ([p (in-list pieces)]) (ref p 'us))))
  (define base-total (layer-median records op (symbol->string loo-base)))
  (define checks (filter-map (lambda (p) (loo-check records op p base-total)) pieces))
  (hasheq 'op op
          'pieces pieces
          'top (and top (symbol->string top))
          'total total
          'sum sum
          'noise (and total sum (positive? total) (/ (- sum total) total))
          'loo checks
          'loo_total base-total
          'contract (contract-cost records op)))

(define (crossing-us records)
  (list-median (map (lambda (r) (ref r 'us_per_crossing)) (kind records "crossing"))))

(define (k1-overhead records)
  (define facade (layer-median records "scalar-add" "facade"))
  (define raw (layer-median records "scalar-add" "raw"))
  (define crossing (crossing-us records))
  (define own (and raw crossing (- raw (* 2 crossing))))
  (hasheq 'facade facade 'libtorch own 'value (and facade own (- facade own))))

(define (k2-pytorch-ratio records)
  (for/list ([op (in-list (map symbol->string pure-overhead-ops))])
    (define r (layer-median records op "facade"))
    (define p (layer-median records op "facade" #:impl "pytorch"))
    (hasheq 'op op 'rktorch r 'pytorch p 'value (and r p (positive? p) (/ r p)))))

(define (impl-rows records impl)
  (filter (lambda (r) (equal? (ref r 'impl) impl)) (kind records "op")))

(define (threads-present records impl)
  (~> (impl-rows records impl)
      (map (lambda (r) (ref r 'threads)) _)
      remove-duplicates
      (sort <)))

(define (ops-of records impl)
  (~> (impl-rows records impl)
      (map (lambda (r) (ref r 'op)) _)
      remove-duplicates))

(define (k3-thread-scaling records #:impl [impl "rktorch"] #:layer [layer "facade"])
  (define ns (filter positive? (threads-present records impl)))
  (define top (and (pair? ns) (last ns)))
  (for*/list ([op (in-list (ops-of records impl))]
              [one (in-value (layer-median records op layer #:impl impl #:threads 1))]
              #:when one)
    (define many (and top (layer-median records op layer #:impl impl #:threads top)))
    (define finalized
      (and top (layer-median records op layer #:impl impl #:threads top #:key 'finalized_per_op)))
    (hasheq 'op op 'threads top 'one one 'many many 'finalized finalized
            'value (and many (positive? one) (/ many one)))))

;; per tensor the op registers, so a chain of ops reads like one op
(define (k4-heap records)
  (for/list ([op (in-list (ops-of records "rktorch"))])
    (define bytes (layer-median records op "facade" #:key 'heap_bytes_per_op))
    (define tensors (counts records op "facade" 'tensors_per_op))
    (hasheq 'op op
            'bytes bytes
            'tensors tensors
            'value (and bytes (real? tensors) (positive? tensors) (/ bytes tensors)))))

(define (pipeline-median records impl pipeline workers key)
  (list-median (for/list ([r (in-list (kind records "pipeline"))]
                          #:when (and (equal? (ref r 'impl) impl)
                                      (equal? (ref r 'pipeline) pipeline)
                                      (equal? (ref r 'workers) workers)
                                      (not (ref r 'error))))
                 (ref r key))))

(define (k5-pipelines records)
  (define pipes (kind records "pipeline"))
  (for*/list ([p (in-list (remove-duplicates (map (lambda (r) (ref r 'pipeline)) pipes)))]
              [w (in-list (~> (map (lambda (r) (ref r 'workers)) pipes) remove-duplicates (sort <)))])
    (define r (pipeline-median records "rktorch" p w 'samples_per_s))
    (define py (pipeline-median records "pytorch" p w 'samples_per_s))
    (hasheq 'pipeline p 'workers w 'rktorch r 'pytorch py
            'value (and r py (positive? py) (/ r py)))))

(define (k6-gc-share records #:pipeline [pipeline "cifar"])
  (define pipes (filter (lambda (r) (and (equal? (ref r 'impl) "rktorch")
                                         (equal? (ref r 'pipeline) pipeline)
                                         (not (ref r 'error))))
                        (kind records "pipeline")))
  (define top (and (pair? pipes) (apply max (map (lambda (r) (ref r 'workers)) pipes))))
  (define shares
    (for/list ([r (in-list pipes)] #:when (equal? (ref r 'workers) top))
      (/ (ref r 'gc_ms) (* 1000.0 (ref r 'elapsed_s)))))
  (hasheq 'pipeline pipeline 'workers top 'value (list-median shares)))

(define (pct x) (if (real? x) (format "~a%" (num (* 100 x) 1)) "-"))

(define (range-text xs)
  (if (pair? xs) (format "~a–~a" (num (apply min xs)) (num (apply max xs))) "-"))

(define (load-range records)
  (define ones (filter-map (lambda (r) (and (pair? (ref r 'load)) (car (ref r 'load))))
                           (append (kind records "op") (kind records "pipeline"))))
  (define busy (filter real? (map (lambda (r) (ref r 'busy_cores)) (kind records "load"))))
  (format "1-min load ~a; busy cores ~a" (range-text ones) (range-text busy)))

(define tolerance-text (pct gap-tolerance))

(define jpeg-order-note
  "jpeg: PyTorch crops uint8 then converts, rktorch converts first (#458)")

(define (k-table records)
  (define overhead (k1-overhead records))
  (define ratios (k2-pytorch-ratio records))
  (define facade-scaling (k3-thread-scaling records))
  (define raw-scaling (k3-thread-scaling records #:layer "raw"))
  (define (scaling-text op)
    (define (of rows) (findf (lambda (x) (equal? (ref x 'op) op)) rows))
    (define t (of facade-scaling))
    (define r (of raw-scaling))
    (format "~a: ~a× at ~a threads (raw layer ~a×; ~a finalized in window)" op
            (num (and t (ref t 'value)) 2) (or (and t (ref t 'threads)) "-")
            (num (and r (ref r 'value)) 2) (pct (and t (ref t 'finalized)))))
  (define heaps (k4-heap records))
  (define pipelines (k5-pipelines records))
  (define gc (k6-gc-share records))
  (table '("#" "metric" "value")
         (list (list "K1" "µs/op, scalar add, 1 thread, minus libtorch's own time"
                     (format "~a (facade ~a, libtorch ~a)" (num (ref overhead 'value) 2)
                             (num (ref overhead 'facade) 2) (num (ref overhead 'libtorch) 2)))
               (list "K2" "rktorch ÷ PyTorch µs/op, 1 thread"
                     (string-join (for/list ([t (in-list ratios)])
                                    (format "~a: ~a× (~a / ~a µs)" (ref t 'op) (num (ref t 'value) 2)
                                            (num (ref t 'rktorch) 2) (num (ref t 'pytorch) 2)))
                                  "; "))
               (list "K3" "per-op cost at 8 threads ÷ 1 thread, with main's in-window finalization"
                     (string-join (map scaling-text (map symbol->string pure-overhead-ops)) "; "))
               (list "K4" "Racket heap bytes per tensor-returning op"
                     (string-join (for/list ([f (in-list heaps)])
                                    (format "~a: ~a~a" (ref f 'op) (whole (ref f 'value))
                                            (if (and (real? (ref f 'tensors)) (= 1 (ref f 'tensors)))
                                                ""
                                                (format " per tensor (~a per op, ~a tensors)"
                                                        (whole (ref f 'bytes))
                                                        (whole (ref f 'tensors))))))
                                  "; "))
               (list "K5" "pipeline samples/s ÷ PyTorch"
                     (if (null? pipelines)
                         "-"
                         (format "~a (~a)"
                                 (string-join (for/list ([f (in-list pipelines)])
                                                (format "~a w=~a: ~a" (ref f 'pipeline)
                                                        (ref f 'workers) (num (ref f 'value) 2)))
                                              "; ")
                                 jpeg-order-note)))
               (list "K6" "GC share of wall time, cifar, most workers"
                     (format "~a at ~a workers" (pct (ref gc 'value)) (or (ref gc 'workers) "-"))))))

(define (attribution-section records op)
  (define a (attribution records op))
  (define rows
    (for/list ([p (in-list (ref a 'pieces))])
      (list (ref p 'piece) (num (ref p 'us) 2) (whole (ref p 'heap)) (num (ref p 'crossings) 1))))
  (define loo-rows
    (for/list ([c (in-list (ref a 'loo))])
      (list (ref c 'piece) (num (ref c 'ladder) 2) (num (ref c 'loo) 2)
            (if (ref c 'agrees) "yes" (format "**no, over ~a**" tolerance-text)))))
  (string-join
   (append
    (list (format "### ~a" op)
          (table '("piece" "µs/op" "heap B/op" "crossings/op")
                 (append
                  rows
                  (list (list (format "median noise (paired steps − ~a's median)"
                                      (or (ref a 'top) "-"))
                              (pct (ref a 'noise)) "" ""))
                  (if (ref a 'contract)
                      (list (list "contract (contracted add − dispatch), beside the ladder"
                                  (num (ref a 'contract) 2) "" ""))
                      '()))))
    (if (null? loo-rows)
        '()
        (list (format "Leave-one-out, against ~a (~a µs/op); agreement within ~a of it:"
                      loo-base (num (ref a 'loo_total) 2) tolerance-text)
              (table '("piece" "ladder µs/op" "leave-one-out µs/op" "agrees") loo-rows))))
   "\n\n"))

(define (thread-section records)
  (define ns (filter positive? (threads-present records "rktorch")))
  (define rows
    (for*/list ([impl (in-list '("rktorch" "pytorch"))]
                [op (in-list (ops-of records impl))]
                [layer (in-list (~> (impl-rows records impl)
                                    (filter (lambda (r) (and (equal? (ref r 'op) op)
                                                             (positive? (ref r 'threads))))
                                            _)
                                    (map (lambda (r) (ref r 'layer)) _)
                                    remove-duplicates))])
      (append (list impl op layer)
              (for/list ([n (in-list (cons 0 ns))])
                (define (m key) (layer-median records op layer #:impl impl #:threads n #:key key))
                (format "~a / ~a / ~a / ~a / ~a / ~a" (num (m 'us_per_op) 2) (whole (m 'ops_per_s))
                        (whole (m 'heap_bytes_per_op)) (whole (m 'gc_ms))
                        (whole (m 'main_cpu_ms)) (pct (m 'finalized_per_op)))))))
  (table (append '("impl" "op" "layer" "main")
                 (map (lambda (n) (format "~a threads" n)) ns))
         rows))

(define sampler-columns '(accounting pressure finalizer shape contract))

(define (sampler-section records)
  (define rows
    (for/list ([r (in-list (kind records "sampler"))])
      (define us (ref r 'us_per_op))
      (append (list (ref r 'op) (ref r 'layer) (num us 2))
              (for/list ([k (in-list sampler-columns)])
                (define share (ref r 'shares k))
                (format "~a (~a µs)" (pct share) (num (and (real? share) (real? us) (* share us)) 2))))))
  (if (null? rows)
      "no sampler records"
      (table (append '("op" "layer" "µs/op") (map symbol->string sampler-columns)) rows)))

(define (error-lines records)
  (for/list ([r (in-list (kind records "pipeline"))] #:when (ref r 'error))
    (format "- ~a w=~a ~a rep ~a: ~a" (ref r 'pipeline) (ref r 'workers) (ref r 'impl)
            (ref r 'rep) (ref r 'error))))

(define (op-cost-markdown records)
  (define meta (findf values (kind records "meta")))
  (define attributed
    (filter (lambda (op) (pair? (present-layers records op))) (ops-of records "rktorch")))
  (define errors (error-lines records))
  (string-join
   (append
    (list "# Per-op cost scoreboard (#255)"
          (if meta (format "~a, ~a; ~a" (commit meta) (host meta) (load-range records)) "")
          (format "One crossing: ~a µs." (num (crossing-us records) 3))
          "## K1–K6"
          (k-table records)
          "## Attribution, one thread")
    (map (lambda (op) (attribution-section records op)) attributed)
    (list "## Threads (µs/op per thread with main's CPU / ops/s summed / heap B/op / GC ms / main CPU ms / finalized in window)"
          (thread-section records)
          "## Sampler (inclusive share of samples, and that share of the sampled µs/op)"
          (sampler-section records))
    (if (null? errors) '() (list "## Pipeline errors" (string-join errors "\n"))))
   "\n\n"
   #:after-last "\n"))
