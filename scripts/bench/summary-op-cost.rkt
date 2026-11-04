#lang racket/base

(require (only-in json read-json)
         (only-in racket/list append-map filter-map last remove-duplicates)
         (only-in racket/string string-join)
         (only-in threading ~>)
         (only-in "harness.rkt" list-median)
         (only-in "markdown.rkt" commit host num ref table whole)
         (only-in "op-cost-layers.rkt" ladder layer-pieces))

(provide read-records
         samples-by-rep
         layer-median
         paired-delta
         piece-label
         attribution
         crossing-us
         k1 k2 k3 k4 k5 k6
         gap-tolerance
         op-cost-markdown)

(define gap-tolerance 0.10)

(define (read-records path)
  (call-with-input-file path
    (lambda (in)
      (for/list ([r (in-port read-json in)] #:when (hash? r)) r))))

(define (kind records k)
  (filter (lambda (r) (equal? (ref r 'kind) k)) records))

(define (op-rows records #:impl impl #:op op #:layer layer #:threads threads)
  (for/list ([r (in-list (kind records "op"))]
             #:when (and (equal? (ref r 'impl) impl) (equal? (ref r 'op) op)
                         (equal? (ref r 'layer) layer) (equal? (ref r 'threads) threads)))
    r))

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

(define (present-layers records op)
  (for/list ([l (in-list ladder)]
             #:when (pair? (op-rows records #:impl "rktorch" #:op op
                                    #:layer (symbol->string l) #:threads 0)))
    l))

(define (counts records op layer key)
  (for/first ([r (in-list (kind records "counts"))]
              #:when (and (equal? (ref r 'op) op) (equal? (ref r 'layer) layer)))
    (ref r key)))

(define (by-rep records op layer key)
  (samples-by-rep (op-rows records #:impl "rktorch" #:op op #:layer (symbol->string layer)
                           #:threads 0)
                  key))

(define (attribution records op)
  (define layers (present-layers records op))
  (define (delta key prev cur)
    (if prev
        (paired-delta (by-rep records op cur key) (by-rep records op prev key))
        (layer-median records op (symbol->string cur) #:key key)))
  (define (crossing-delta prev cur)
    (define c (counts records op (symbol->string cur) 'crossings_per_op))
    (define p (and prev (counts records op (symbol->string prev) 'crossings_per_op)))
    (and c (if prev (and p (- c p)) c)))
  (define pieces
    (for/list ([cur (in-list layers)]
               [prev (in-list (cons #f layers))])
      (define spanned
        (for/list ([l (in-list (member cur (reverse ladder)))]
                   #:break (eq? l prev))
          l))
      (hasheq 'piece (piece-label (reverse spanned))
              'layer (symbol->string cur)
              'us (delta 'us_per_op prev cur)
              'heap (delta 'heap_bytes_per_op prev cur)
              'crossings (crossing-delta prev cur))))
  (define top (and (pair? layers) (last layers)))
  (define total (and top (layer-median records op (symbol->string top))))
  (define sum (and (pair? pieces) (andmap (lambda (p) (real? (ref p 'us))) pieces)
                   (for/sum ([p (in-list pieces)]) (ref p 'us))))
  (define gap (and total sum (positive? total) (/ (- total sum) total)))
  (hasheq 'op op
          'pieces pieces
          'top (and top (symbol->string top))
          'total total
          'sum sum
          'gap gap
          'within (and gap (<= (abs gap) gap-tolerance))
          'contract (let ([u (by-rep records op 'unchecked 'us_per_op)])
                      (and (positive? (hash-count u))
                           (~> (op-rows records
                                        #:impl "rktorch"
                                        #:op op
                                        #:layer "contracted"
                                        #:threads 0)
                               (samples-by-rep 'us_per_op)
                               (paired-delta u))))))

(define (crossing-us records)
  (list-median (map (lambda (r) (ref r 'us_per_crossing)) (kind records "crossing"))))

(define (k1 records)
  (define facade (layer-median records "scalar-add" "facade"))
  (define raw (layer-median records "scalar-add" "raw"))
  (define crossing (crossing-us records))
  (define own (and raw crossing (- raw (* 2 crossing))))
  (hasheq 'facade facade 'libtorch own 'value (and facade own (- facade own))))

(define (k2 records)
  (for/list ([op (in-list '("scalar-add" "add-8x8"))])
    (define r (layer-median records op "facade"))
    (define p (layer-median records op "facade" #:impl "pytorch"))
    (hasheq 'op op 'rktorch r 'pytorch p 'value (and r p (positive? p) (/ r p)))))

(define (threads-present records impl)
  (~> (kind records "op")
      (filter (lambda (r) (equal? (ref r 'impl) impl)) _)
      (map (lambda (r) (ref r 'threads)) _)
      remove-duplicates
      (sort <)))

(define (k3 records #:impl [impl "rktorch"] #:layer [layer "facade"])
  (define ns (filter positive? (threads-present records impl)))
  (define top (and (pair? ns) (last ns)))
  (for*/list ([op (in-list (ops-of records impl))]
              [one (in-value (layer-median records op layer #:impl impl #:threads 1))]
              #:when one)
    (define many (and top (layer-median records op layer #:impl impl #:threads top)))
    (hasheq 'op op 'threads top 'one one 'many many
            'value (and many (positive? one) (/ many one)))))

(define (ops-of records impl)
  (~> (kind records "op")
      (filter (lambda (r) (equal? (ref r 'impl) impl)) _)
      (map (lambda (r) (ref r 'op)) _)
      remove-duplicates))

(define (k4 records)
  (for/list ([op (in-list (ops-of records "rktorch"))])
    (hasheq 'op op
            'value (layer-median records op "facade" #:key 'heap_bytes_per_op)
            'tensors (counts records op "facade" 'tensors_per_op))))

(define (pipeline-median records impl pipeline workers key)
  (list-median (for/list ([r (in-list (kind records "pipeline"))]
                          #:when (and (equal? (ref r 'impl) impl)
                                      (equal? (ref r 'pipeline) pipeline)
                                      (equal? (ref r 'workers) workers)
                                      (not (ref r 'error))))
                 (ref r key))))

(define (k5 records)
  (define pipes (kind records "pipeline"))
  (for*/list ([p (in-list (remove-duplicates (map (lambda (r) (ref r 'pipeline)) pipes)))]
              [w (in-list (~> (map (lambda (r) (ref r 'workers)) pipes) remove-duplicates (sort <)))])
    (define r (pipeline-median records "rktorch" p w 'samples_per_s))
    (define py (pipeline-median records "pytorch" p w 'samples_per_s))
    (hasheq 'pipeline p 'workers w 'rktorch r 'pytorch py
            'value (and r py (positive? py) (/ r py)))))

(define (k6 records #:pipeline [pipeline "cifar"])
  (define pipes (filter (lambda (r) (and (equal? (ref r 'impl) "rktorch")
                                         (equal? (ref r 'pipeline) pipeline)
                                         (not (ref r 'error))))
                        (kind records "pipeline")))
  (define top (for/fold ([m #f]) ([r (in-list pipes)]) (max (or m 0) (ref r 'workers))))
  (define shares
    (for/list ([r (in-list pipes)] #:when (equal? (ref r 'workers) top))
      (/ (ref r 'gc_ms) (* 1000.0 (ref r 'elapsed_s)))))
  (hasheq 'pipeline pipeline 'workers top 'value (list-median shares)))

(define (pct x) (if (real? x) (format "~a%" (num (* 100 x) 1)) "-"))

(define (load-range records)
  (define ones (filter-map (lambda (r) (and (pair? (ref r 'load)) (car (ref r 'load))))
                           (append (kind records "op") (kind records "pipeline"))))
  (define busy (filter real? (map (lambda (r) (ref r 'busy_cores)) (kind records "load"))))
  (format "1-min load ~a–~a; busy cores ~a–~a"
          (if (pair? ones) (num (apply min ones)) "-") (if (pair? ones) (num (apply max ones)) "-")
          (if (pair? busy) (num (apply min busy)) "-") (if (pair? busy) (num (apply max busy)) "-")))

(define (k-table records)
  (define one (k1 records))
  (define twos (k2 records))
  (define threes (k3 records))
  (define raws (k3 records #:layer "raw"))
  (define (three-text op)
    (define t (for/first ([x (in-list threes)] #:when (equal? (ref x 'op) op)) x))
    (define r (for/first ([x (in-list raws)] #:when (equal? (ref x 'op) op)) x))
    (format "~a: ~a× at ~a threads (raw layer ~a×)" op (num (and t (ref t 'value)) 2)
            (or (and t (ref t 'threads)) "-") (num (and r (ref r 'value)) 2)))
  (define fours (k4 records))
  (define fives (k5 records))
  (define six (k6 records))
  (table '("#" "metric" "value")
         (list (list "K1" "µs/op, scalar add, 1 thread, minus libtorch's own time"
                     (format "~a (facade ~a, libtorch ~a)" (num (ref one 'value) 2)
                             (num (ref one 'facade) 2) (num (ref one 'libtorch) 2)))
               (list "K2" "rktorch ÷ PyTorch µs/op, 1 thread"
                     (string-join (for/list ([t (in-list twos)])
                                    (format "~a: ~a× (~a / ~a µs)" (ref t 'op) (num (ref t 'value) 2)
                                            (num (ref t 'rktorch) 2) (num (ref t 'pytorch) 2)))
                                  "; "))
               (list "K3" "per-op cost at 8 threads ÷ 1 thread"
                     (string-join (map three-text '("scalar-add" "add-8x8")) "; "))
               (list "K4" "Racket heap bytes per tensor-returning op"
                     (string-join (for/list ([f (in-list fours)])
                                    (format "~a: ~a" (ref f 'op) (whole (ref f 'value))))
                                  "; "))
               (list "K5" "pipeline samples/s ÷ PyTorch"
                     (if (null? fives)
                         "-"
                         (string-join (for/list ([f (in-list fives)])
                                        (format "~a w=~a: ~a" (ref f 'pipeline) (ref f 'workers)
                                                (num (ref f 'value) 2)))
                                      "; ")))
               (list "K6" "GC share of wall time, cifar, most workers"
                     (format "~a at ~a workers" (pct (ref six 'value)) (or (ref six 'workers) "-"))))))

(define (attribution-section records op)
  (define a (attribution records op))
  (define rows
    (for/list ([p (in-list (ref a 'pieces))])
      (list (ref p 'piece) (num (ref p 'us) 2) (whole (ref p 'heap)) (num (ref p 'crossings) 1))))
  (format
   "### ~a\n\n~a"
   op
   (table '("piece" "µs/op" "heap B/op" "crossings/op")
          (append
           rows
           (list (list "**sum of pieces**" (num (ref a 'sum) 2) "" "")
                 (list (format "**total (~a)**" (or (ref a 'top) "-")) (num (ref a 'total) 2) "" "")
                 (list (if (ref a 'within) "gap" "**gap, over 10%**") (pct (ref a 'gap)) "" ""))
           (if (ref a 'contract)
               (list (list "contract (contracted add − dispatch), not in the sum"
                           (num (ref a 'contract) 2)
                           ""
                           ""))
               '())))))

(define (thread-section records)
  (define rows
    (for*/list ([impl (in-list '("rktorch" "pytorch"))]
                [op (in-list (ops-of records impl))]
                [layer (in-list (remove-duplicates
                                 (for/list ([r (in-list (kind records "op"))]
                                            #:when (and (equal? (ref r 'impl) impl)
                                                        (equal? (ref r 'op) op)
                                                        (positive? (ref r 'threads))))
                                   (ref r 'layer))))])
      (define ns (filter positive? (threads-present records impl)))
      (append (list impl op layer)
              (for/list ([n (in-list (cons 0 ns))])
                (define (m key) (layer-median records op layer #:impl impl #:threads n #:key key))
                (format "~a / ~a / ~a / ~a" (num (m 'us_per_op) 2) (whole (m 'ops_per_s))
                        (whole (m 'heap_bytes_per_op)) (whole (m 'gc_ms)))))))
  (define ns (filter positive? (threads-present records "rktorch")))
  (table (append '("impl" "op" "layer" "main")
                 (map (lambda (n) (format "~a threads" n)) ns))
         rows))

(define (sampler-section records)
  (define rows
    (for/list ([r (in-list (kind records "sampler"))])
      (define shares (ref r 'shares))
      (cons (ref r 'op)
            (for/list ([k (in-list '(accounting pressure finalizer shape contract))])
              (pct (ref shares k))))))
  (if (null? rows)
      "no sampler records"
      (table '("op" "accounting" "pressure" "finalizer" "shape" "contract") rows)))

(define (error-lines records)
  (for/list ([r (in-list (kind records "pipeline"))] #:when (ref r 'error))
    (format "- ~a w=~a ~a rep ~a: ~a" (ref r 'pipeline) (ref r 'workers) (ref r 'impl)
            (ref r 'rep) (ref r 'error))))

(define (op-cost-markdown records)
  (define meta (for/first ([r (in-list (kind records "meta"))]) r))
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
    (list "## Threads (µs/op per thread / ops/s summed / heap B/op / GC ms)"
          (thread-section records)
          "## Sampler (inclusive share of samples)"
          (sampler-section records))
    (if (null? errors) '() (list "## Pipeline errors" (string-join errors "\n"))))
   "\n\n"
   #:after-last "\n"))
