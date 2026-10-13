#lang racket/base

(require (only-in json read-json)
         (only-in racket/format ~a ~r)
         (only-in racket/list append* append-map remove-duplicates)
         (only-in racket/match match-define)
         (only-in racket/string string-join string-trim)
         (only-in "harness.rkt" paired-ratio)
         (only-in "layers.rkt" layer-order))

(provide read-records
         summary-markdown)

(define (records-in in)
  (for/list ([r (in-port read-json in)] #:when (hash? r))
    r))

(define (read-records paths)
  (append-map (lambda (p) (call-with-input-file p records-in)) paths))

(define (ref r . keys)
  (for/fold ([v r]) ([k (in-list keys)])
    (and (hash? v) (hash-ref v k #f))))

(define (num x [digits 1])
  (if (real? x) (~r x #:precision (list '= digits)) "-"))

(define (load-text r)
  (define l (ref r 'load))
  (if (pair? l) (num (car l)) "-"))

(define (commit r)
  (define sha (ref r 'meta 'git_sha))
  (format "`~a`~a"
          (if (string? sha) (substring sha 0 (min 7 (string-length sha))) "?")
          (if (ref r 'meta 'git_dirty) " (dirty)" "")))

(define (host r)
  (define m (ref r 'meta))
  (format "~a (~a, ~a, cpus ~a)" (ref m 'host) (ref m 'cpu) (ref m 'os) (ref m 'affinity)))

(define (execution r)
  (define m (ref r 'meta))
  (format "~a, ~a threads (~a)~a"
          (ref m 'device) (ref m 'threads) (ref m 'threads_source)
          (if (string? (ref m 'gpu)) (format ", ~a" (ref m 'gpu)) "")))

(define (invocation r)
  (format "~a, ~a, ~a, run ~a" (commit r) (host r) (execution r) (ref r 'run_id)))

(define (one-line text)
  (regexp-replace* #rx"[ \t]*[\r\n]+[ \t]*" (string-trim (~a text)) " / "))

(define (cell text)
  (regexp-replace* #rx"\\|" (one-line text) "\\\\|"))

(define (table header rows)
  (string-join
   (append (list (string-join header " | " #:before-first "| " #:after-last " |")
                 (string-join (map (lambda (_) "---") header)
                              "|"
                              #:before-first "|"
                              #:after-last "|"))
           (for/list ([row (in-list rows)])
             (string-join (map cell row) " | " #:before-first "| " #:after-last " |")))
   "\n"))

(define (median r) (ref r 'stats 'median))
(define (iqr r) (ref r 'stats 'iqr))

(define (select records suite group)
  (for/list ([r (in-list records)]
             #:when (and (equal? (ref r 'suite) suite) (equal? (ref r 'group) group)))
    r))

(define (by-context records context)
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

(define (ratio-row example base other)
  (define cmp (paired-ratio (ref base 'samples) (ref other 'samples)))
  (define n (hash-ref cmp 'n))
  (list example
        (format "~a vs ~a" (or (ref other 'variant) "-") (or (ref base 'variant) "-"))
        (num (hash-ref cmp 'median) 3)
        (format "~a-~a" (num (hash-ref cmp 'lo) 3) (num (hash-ref cmp 'hi) 3))
        (format "~a%~a" (~r (* 100 (hash-ref cmp 'confidence)) #:precision 0) (if (< n 6) " (n < 6)" ""))
        n))

(define (ratio-rows records)
  (append*
   (for/list ([g (in-list (by-context records (lambda (r) (list (ref r 'case) (ref r 'run_id)))))])
     (match-define (list* (list example _) base others) g)
     (for/list ([other (in-list others)]
                #:when (equal? (length (ref base 'samples)) (length (ref other 'samples))))
       (ratio-row example base other)))))

(define (settings-text settings)
  (if (hash? settings)
      (string-join (for/list ([k (in-list (sort (hash-keys settings) symbol<?))]
                              #:unless (regexp-match? #rx"^RKTORCH_.*_DIR$" (symbol->string k)))
                     (format "~a=~a" k (hash-ref settings k)))
                   " ")
      "-"))

(define (e2e-section records)
  (define runs
    (table '("example" "variant" "commit" "execution" "settings" "unit" "median" "IQR"
             "steps/s" "runs" "load")
           (for/list ([r (in-list records)])
             (list (ref r 'case) (or (ref r 'variant) "-") (commit r) (execution r)
                   (settings-text (ref r 'settings))
                   (ref r 'unit) (num (median r) 2) (num (iqr r) 2)
                   (num (ref r 'steps_per_s 'median) 2) (ref r 'stats 'n) (load-text r)))))
  (define ratios (ratio-rows records))
  (if (pair? ratios)
      (string-append runs "\n\n"
                     (table '("example" "time ratio" "median" "interval" "confidence" "rounds")
                            ratios))
      runs))

(define (failure-section records)
  (table '("example" "variant" "commit" "execution" "exit code" "error")
         (for/list ([r (in-list records)])
           (list (ref r 'case) (or (ref r 'variant) "-") (commit r) (execution r)
                 (or (ref r 'exit_code) "-") (or (ref r 'error) "-")))))

(struct section (suite group title render context))

(define sections
  (list (section "micro" "crossings" "Native crossings by signature class"
                 (plain-section "crossing") invocation)
        (section "micro" "layers" "The op stack, layer by layer" layer-section invocation)
        (section "micro" "ops" "Per op" op-section invocation)
        (section "micro" "contracts" "Validation strategies" contract-section invocation)
        (section "micro" "pipeline" "A real pipeline" (plain-section "pipeline") invocation)
        (section "e2e" "summary" "End to end" e2e-section host)
        (section "e2e" "error" "Failures" failure-section host)))

(define (header records)
  (define (all key) (string-join (remove-duplicates (map (lambda (r) (~a (ref r 'meta key))) records))
                                 " / "))
  (format "Commits ~a; Racket ~a, libtorch ~a, ~a (~a cores)."
          (string-join (remove-duplicates (map commit records)) ", ")
          (all 'racket) (all 'libtorch) (all 'cpu) (all 'cores)))

(define (summary-markdown records #:title [title "Benchmarks"])
  (string-join
   (append
    (list (format "# ~a" title) "" (if (pair? records) (header records) "No records.") "")
    (for*/list ([s (in-list sections)]
                [rs (in-value (select records (section-suite s) (section-group s)))]
                #:when (pair? rs)
                [c (in-list (by-context rs (section-context s)))])
      (string-append "## " (section-title s) " (" (one-line (car c)) ")\n\n"
                     ((section-render s) (cdr c))
                     "\n")))
   "\n"))

(module+ main
  (require (only-in racket/cmdline command-line))
  (define title "Benchmarks")
  (define paths
    (command-line
     #:once-each [("--title") t "Heading" (set! title t)]
     #:args paths paths))
  (display (summary-markdown (read-records paths) #:title title)))

(module+ test
  (require (only-in json write-json)
           (only-in racket/file make-temporary-file)
           rackunit
           (only-in "harness.rkt" with-temporary-path))

  (define (meta host #:threads [threads 4] #:source [source "default"])
    (hasheq 'git_sha "abcdef0123" 'device "cpu" 'threads threads 'threads_source source
            'host host 'cpu "Some CPU" 'os "linux-x86_64" 'affinity "0-3"))

  (define (summary-record variant run-id host samples #:meta [m (meta host)])
    (hasheq 'suite "e2e" 'group "summary" 'case "stepper" 'variant variant 'unit "ms/step"
            'run_id run-id 'samples samples 'stats (hasheq 'median (car samples) 'n 2)
            'meta m))

  (define (crossing-record run-id median)
    (hasheq 'suite "micro" 'group "crossings" 'case "scalar" 'unit "ns" 'run_id run-id
            'stats (hasheq 'median median 'iqr 1.0) 'meta (meta "lab")))

  (test-case "ratios pair variants within one run, never across runs"
    (define md
      (summary-markdown
       (list (summary-record "a" "r1" "lab" '(10.0 10.0))
             (summary-record "b" "r1" "lab" '(5.0 5.0))
             (summary-record "a" "r2" "lab" '(20.0 20.0))
             (summary-record "b" "r2" "lab" '(10.0 10.0)))))
    (check-equal? (length (regexp-match* #rx"\\| stepper \\| b vs a \\| 0\\.500 \\|" md)) 2)
    (check-false (regexp-match? #rx"a vs a|b vs b|a vs b" md)))

  (test-case "the settings cell leaves the dataset caches out, in key order"
    (define md
      (summary-markdown
       (list (hash-set (summary-record "a" "r1" "lab" '(10.0 10.0))
                       'settings (hasheq 'STEPS "300" 'RKTORCH_MNIST_DIR "/c/mnist"
                                         'EPOCHS "3" 'OMP_NUM_THREADS "1")))))
    (check-regexp-match #rx"\\| EPOCHS=3 OMP_NUM_THREADS=1 STEPS=300 \\|" md)
    (check-false (regexp-match? #rx"RKTORCH_MNIST_DIR" md)))

  (test-case "e2e sections split by host"
    (define md
      (summary-markdown
       (list (summary-record "a" "r1" "lab" '(10.0 10.0))
             (summary-record "b" "r2" "laptop" '(5.0 5.0)))))
    (check-equal? (length (regexp-match* #rx"## End to end" md)) 2)
    (check-regexp-match #rx"lab \\(Some CPU, linux-x86_64, cpus 0-3\\)" md)
    (check-false (regexp-match? #rx"vs" md)))

  (test-case "an A/B over OMP_NUM_THREADS shares a section and yields a ratio"
    (define md
      (summary-markdown
       (list (summary-record "one" "r1" "lab" '(20.0 20.0)
                             #:meta (meta "lab" #:threads 1 #:source "OMP_NUM_THREADS"))
             (summary-record "four" "r1" "lab" '(5.0 5.0)
                             #:meta (meta "lab" #:threads 4 #:source "OMP_NUM_THREADS")))))
    (check-equal? (length (regexp-match* #rx"## End to end" md)) 1)
    (check-regexp-match #rx"\\| stepper \\| four vs one \\| 0\\.250 \\|" md)
    (check-regexp-match #rx"cpu, 1 threads \\(OMP_NUM_THREADS\\)" md)
    (check-regexp-match #rx"cpu, 4 threads \\(OMP_NUM_THREADS\\)" md))

  (define (error-record variant code message)
    (hasheq 'suite "e2e" 'group "error" 'case "stepper" 'variant variant 'exit_code code
            'error message 'run_id "r1" 'meta (meta "lab")))

  (define (markdown-of records)
    (with-temporary-path (path (make-temporary-file "summary-~a.jsonl"))
      (call-with-output-file path #:exists 'truncate
        (lambda (out) (for ([r (in-list records)]) (write-json r out) (newline out))))
      (summary-markdown (read-records (list path)))))

  (test-case "a file of failures alone still reports them"
    (define md (markdown-of (list (error-record "b" 1 "e2e: stepper: boom (exit code 1)")
                                  (hasheq 'suite "e2e" 'group "error" 'case "06-gpt"
                                          'error "no example" 'meta (meta "lab")))))
    (check-false (regexp-match? #rx"## End to end" md))
    (check-regexp-match #rx"## Failures \\(lab " md)
    (check-regexp-match
     #rx"\\| stepper \\| b \\| `abcdef0` \\| cpu, 4 threads \\(default\\) \\| 1 \\| e2e: " md)
    (check-regexp-match #rx"\\| 06-gpt \\| - \\| `abcdef0` \\| [^|]* \\| - \\| no example \\|" md))

  (test-case "failures follow the runs that succeeded"
    (define md (markdown-of (list (summary-record "a" "r1" "lab" '(10.0 10.0))
                                  (error-record "b" 137 "killed\nby | signal"))))
    (check-regexp-match #rx"## End to end[^#]*\\| stepper \\| a \\|[^#]*## Failures" md)
    (check-regexp-match #rx"\\| b \\| [^|]* \\| [^|]* \\| 137 \\| killed / by \\\\\\| signal \\|\n"
                        md))

  (test-case "each failure names the build and the execution that produced it"
    (define (failure variant sha device)
      (hash-set* (error-record variant 1 "boom")
                 'meta (hash-set* (meta "lab") 'git_sha sha 'device device)))
    (define md (markdown-of (list (failure "old" "1111111aaa" "cpu")
                                  (failure "new" "2222222bbb" "cuda"))))
    (check-equal? (length (regexp-match* #rx"## Failures" md)) 1)
    (check-regexp-match #rx"\\| stepper \\| old \\| `1111111` \\| cpu, 4 threads " md)
    (check-regexp-match #rx"\\| stepper \\| new \\| `2222222` \\| cuda, 4 threads " md))

  (test-case "a multi-line GPU name stays in its cell"
    (define m (hash-set* (meta "lab") 'device "cuda" 'gpu "GPU A\nGPU B"))
    (define md (summary-markdown (list (summary-record "a" "r1" "lab" '(1.0 1.0) #:meta m))))
    (check-regexp-match #rx"cuda, 4 threads \\(default\\), GPU A / GPU B \\|" md))

  (test-case "appended micro invocations keep a section each"
    (define md (summary-markdown (list (crossing-record "r1" 100.0) (crossing-record "r2" 200.0))))
    (check-equal? (length (regexp-match* #rx"## Native crossings" md)) 2)
    (check-regexp-match #rx"run r1\\)\n\n[^#]*\\| scalar \\| 100\\.0 \\|" md)
    (check-regexp-match #rx"run r2\\)\n\n[^#]*\\| scalar \\| 200\\.0 \\|" md)))
