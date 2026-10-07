#lang racket/base

(require (only-in json read-json)
         (only-in racket/format ~a ~r)
         (only-in racket/list remove-duplicates)
         (only-in racket/string string-join)
         (only-in "layers.rkt" layer-order))

(provide read-records
         summary-markdown)

(define (read-records paths)
  (for*/list ([p (in-list paths)]
              [r (in-port read-json (open-input-file p))]
              #:when (hash? r))
    r))

(define (ref r . keys)
  (for/fold ([v r]) ([k (in-list keys)])
    (and (hash? v) (hash-ref v k #f))))

(define (num x [digits 1])
  (if (real? x) (~r x #:precision (list '= digits)) "-"))

(define (load-text r)
  (define l (ref r 'load))
  (if (pair? l) (num (car l)) "-"))

(define (context r)
  (define m (ref r 'meta))
  (format "~a, ~a threads (~a)~a"
          (ref m 'device) (ref m 'threads) (ref m 'threads_source)
          (if (string? (ref m 'gpu)) (format ", ~a" (ref m 'gpu)) "")))

(define (table header rows)
  (string-join
   (append (list (string-join header " | " #:before-first "| " #:after-last " |")
                 (string-join (map (lambda (_) "---") header)
                              "|"
                              #:before-first "|"
                              #:after-last "|"))
           (for/list ([row (in-list rows)])
             (string-join (map ~a row) " | " #:before-first "| " #:after-last " |")))
   "\n"))

(define (median r) (ref r 'stats 'median))
(define (iqr r) (ref r 'stats 'iqr))

(define (select records suite group)
  (for/list ([r (in-list records)]
             #:when (and (equal? (ref r 'suite) suite) (equal? (ref r 'group) group)))
    r))

(define (by-context records)
  (for/list ([c (in-list (remove-duplicates (map context records)))])
    (cons c (filter (lambda (r) (equal? (context r) c)) records))))

(define ((plain-section label) records)
  (table (list label (ref (car records) 'unit) "IQR" "load")
         (for/list ([r (in-list records)])
           (list (ref r 'case) (num (median r)) (num (iqr r)) (load-text r)))))

(define (layer-section records)
  (define (find name) (for/first ([r (in-list records)] #:when (equal? (ref r 'case) name)) r))
  (define ladder (filter values (map (lambda (s) (find (symbol->string s))) layer-order)))
  (define rows
    (for/list ([r (in-list ladder)]
               [prev (in-list (cons #f ladder))])
      (list (ref r 'case) (num (/ (median r) 1000.0) 2)
            (if prev (num (/ (- (median r) (median prev)) 1000.0) 2) "-")
            (num (/ (iqr r) 1000.0) 2) (load-text r))))
  (define fin (find "finalization"))
  (table '("layer (8x8 add)" "µs/call" "added" "IQR" "load")
         (if fin
             (append rows (list (list "finalization (drain)" (num (/ (median fin) 1000.0) 2)
                                      "-" (num (/ (iqr fin) 1000.0) 2) (load-text fin))))
             rows)))

(define (op-section records)
  (define ops (remove-duplicates (map (lambda (r) (ref r 'case)) records)))
  (define (cell op variant)
    (for/first ([r (in-list records)]
                #:when (and (equal? (ref r 'case) op) (equal? (ref r 'variant) variant)))
      r))
  (table '("op" "facade µs" "unchecked µs" "raw µs" "facade/raw" "load")
         (for/list ([op (in-list ops)])
           (define f (cell op "facade"))
           (define u (cell op "unchecked"))
           (define w (cell op "raw"))
           (define (us r) (if r (num (/ (median r) 1000.0) 2) "-"))
           (list op (us f) (us u) (us w)
                 (if (and f w) (num (/ (median f) (median w)) 2) "-")
                 (load-text (or f u w))))))

(define (contract-section records)
  (define workloads (remove-duplicates (map (lambda (r) (ref r 'case)) records)))
  (table '("workload" "strategy" "intra ns" "cross ns" "cross vs bare")
         (for*/list ([w (in-list workloads)]
                     [rows (in-value (filter (lambda (r) (equal? (ref r 'case) w)) records))]
                     [base (in-value (for/first ([r (in-list rows)]
                                                 #:when (equal? (ref r 'variant) "bare/cross"))
                                       (median r)))]
                     [s (in-list '("bare" "unless+error" "define/contract" "contract-out"))])
           (define (at where)
             (for/first ([r (in-list rows)]
                         #:when (equal? (ref r 'variant) (string-append s "/" where)))
               (median r)))
           (list w s (num (at "intra")) (num (at "cross"))
                 (if (and base (at "cross")) (num (/ (at "cross") base) 2) "-")))))

(define (e2e-section records)
  (table '("example" "variant" "settings" "unit" "median" "IQR" "steps/s" "runs" "load")
         (for/list ([r (in-list records)])
           (define settings (ref r 'settings))
           (list (ref r 'case) (or (ref r 'variant) "-")
                 (if (hash? settings)
                     (string-join (for/list ([(k v) (in-hash settings)]) (format "~a=~a" k v)) " ")
                     "-")
                 (ref r 'unit) (num (median r) 2) (num (iqr r) 2)
                 (num (ref r 'steps_per_s 'median) 2) (ref r 'stats 'n) (load-text r)))))

(define sections
  (list (list "micro" "crossings" "Native crossings by signature class" (plain-section "crossing"))
        (list "micro" "layers" "The op stack, layer by layer" layer-section)
        (list "micro" "ops" "Per op" op-section)
        (list "micro" "contracts" "Validation strategies" contract-section)
        (list "micro" "pipeline" "A real pipeline" (plain-section "pipeline"))
        (list "e2e" "summary" "End to end" e2e-section)))

(define (header records)
  (define m (ref (car records) 'meta))
  (format "Commit `~a`~a, Racket ~a, libtorch ~a, ~a (~a cores)."
          (let ([sha (ref m 'git_sha)]) (if (string? sha) (substring sha 0 (min 7 (string-length sha))) "?"))
          (if (ref m 'git_dirty) " (dirty)" "")
          (ref m 'racket) (ref m 'libtorch) (ref m 'cpu) (ref m 'cores)))

(define (summary-markdown records #:title [title "Benchmarks"])
  (string-join
   (append
    (list (format "# ~a" title) "" (if (pair? records) (header records) "No records.") "")
    (for*/list ([s (in-list sections)]
                [rs (in-value (select records (car s) (cadr s)))]
                #:when (pair? rs)
                [c (in-list (by-context rs))])
      (string-append "## " (caddr s) " (" (car c) ")\n\n" ((cadddr s) (cdr c)) "\n")))
   "\n"))

(module+ main
  (require (only-in racket/cmdline command-line))
  (define title "Benchmarks")
  (define paths
    (command-line
     #:once-each [("--title") t "Heading" (set! title t)]
     #:args paths paths))
  (display (summary-markdown (read-records paths) #:title title)))
