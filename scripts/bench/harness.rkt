#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in json jsexpr->string)
         (only-in racket/file delete-directory/files)
         (only-in racket/format ~a ~r)
         (only-in racket/future processor-count)
         (only-in racket/list drop filter-map remove-duplicates take)
         (only-in racket/match match match-define)
         (only-in racket/port open-output-nowhere port->string with-output-to-string)
         (only-in racket/string string-prefix? string-split string-trim)
         (only-in racket/system process*/ports)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in threading ~>))

(provide bench-case
         bench-case?
         bench-case-before
         bench-case-name
         (struct-out result)
         measure
         clock-ms
         quantile
         summarize
         mean
         list-median
         paired-ratio
         rotate
         parse-load-average
         load-average
         parse-cpu-times
         busy-cores
         thread-setting
         cuda-gpu-name
         with-temporary-path
         host-meta
         result->record
         write-record
         format-result)

(struct bench-case (name run reps before self-timed?)
  #:constructor-name make-bench-case
  #:omit-define-syntaxes)

(define (bench-case name run ;; noqa
                    #:reps [reps 1]
                    #:before [before void]
                    #:self-timed? [self-timed? #f])
  (make-bench-case name run reps before self-timed?))

(struct result (name reps per-call gc-ms load) #:transparent)

(define (rotate xs k)
  (define n (length xs))
  (cond
    [(zero? n) xs]
    [else
     (define m (modulo k n))
     (append (drop xs m) (take xs m))]))

(define (clock-ms) (current-inexact-monotonic-milliseconds))

(define (run-once c)
  ((bench-case-before c))
  (define gc0 (current-gc-milliseconds))
  (define t0 (clock-ms))
  (define answer ((bench-case-run c) (bench-case-reps c)))
  (define t1 (clock-ms))
  (define gc1 (current-gc-milliseconds))
  (define elapsed (if (bench-case-self-timed? c) answer (- t1 t0)))
  (cons (/ elapsed (bench-case-reps c)) (- gc1 gc0)))

(define (measure cases
                 #:warmup [warmup 1]
                 #:rounds [rounds 5]
                 #:load [load load-average]
                 #:on-failure [on-failure #f])
  (define failed (make-hasheq))
  (define (attempt c)
    (cond
      [(hash-ref failed c #f) #f]
      [on-failure
       (with-handlers ([exn:fail? (lambda (e) (hash-set! failed c #t) (on-failure e) #f)])
         (run-once c))]
      [else (run-once c)]))
  (for* ([_ (in-range warmup)] [c (in-list cases)])
    (attempt c))
  (define samples (make-hasheq))
  (define start-load (load))
  (for* ([i (in-range rounds)]
         [c (in-list (rotate cases i))]
         [sample (in-value (attempt c))]
         #:when sample)
    (hash-update! samples c (lambda (acc) (cons sample acc)) '()))
  (define end-load (load))
  (for/list ([c (in-list cases)])
    (define rows (reverse (hash-ref samples c '())))
    (result (bench-case-name c) (bench-case-reps c)
            (map car rows) (map cdr rows)
            (list start-load end-load))))

(define (quantile xs p)
  (define v (list->vector (sort xs <)))
  (define n (vector-length v))
  (define h (* (sub1 n) p))
  (define lo (inexact->exact (floor h)))
  (define hi (min (sub1 n) (add1 lo)))
  (define frac (- h lo))
  (+ (vector-ref v lo) (* frac (- (vector-ref v hi) (vector-ref v lo)))))

(define (mean xs)
  (and (pair? xs) (/ (apply + xs) (length xs) 1.0)))

(define (list-median xs)
  (define reals (if (list? xs) (filter real? xs) '()))
  (and (pair? reals) (quantile reals 0.5)))

(define (summarize xs)
  (define q1 (quantile xs 0.25))
  (define q3 (quantile xs 0.75))
  (hasheq 'median (exact->inexact (quantile xs 0.5))
          'q1 (exact->inexact q1)
          'q3 (exact->inexact q3)
          'iqr (exact->inexact (- q3 q1))
          'min (exact->inexact (apply min xs))
          'max (exact->inexact (apply max xs))
          'n (length xs)))

(define (choose n k)
  (for/fold ([acc 1]) ([i (in-range k)])
    (/ (* acc (- n i)) (add1 i))))

(define (below-tail n k)
  (for/sum ([i (in-range k)]) (/ (choose n i) (expt 2 n))))

(define (paired-ratio a-samples b-samples #:level [level 0.95])
  (define ratios (sort (map / b-samples a-samples) <))
  (define n (length ratios))
  (define alpha/2 (/ (- 1 level) 2))
  (define k
    (for/last ([k (in-range 1 (add1 (quotient n 2)))]
               #:when (<= (below-tail n k) alpha/2))
      k))
  (define rank (or k 1))
  (hasheq 'median (exact->inexact (quantile ratios 0.5))
          'lo (exact->inexact (list-ref ratios (sub1 rank)))
          'hi (exact->inexact (list-ref ratios (- n rank)))
          'confidence (exact->inexact (- 1 (* 2 (below-tail n rank))))
          'n n))

(define (parse-load-average text)
  (define numbers
    (for*/list ([w (in-list (string-split text))]
                [x (in-value (string->number w))]
                #:when (real? x))
      (exact->inexact x)))
  (and (>= (length numbers) 3) (take numbers 3)))

(define (command-output program . args)
  (define path (find-executable-path program))
  (cond
    [(not path) #f]
    [else
     (match-define (list out in _pid _err control)
       (apply process*/ports #f #f (open-output-nowhere) path args))
     (close-output-port in)
     (define text (port->string out))
     (close-input-port out)
     (control 'wait)
     (and (zero? (control 'exit-code)) (string-trim text))]))

(define (read-first-line path)
  (and (file-exists? path)
       (call-with-input-file path read-line)))

(define (load-average)
  (define text
    (or (read-first-line "/proc/loadavg")
        (command-output "sysctl" "-n" "vm.loadavg")))
  (and (string? text) (parse-load-average text)))

(define (proc-field path key)
  (define pattern (regexp (string-append "^" (regexp-quote key) "[^:]*:(.*)$")))
  (and (file-exists? path)
       (call-with-input-file path
         (lambda (in)
           (for/or ([line (in-lines in)])
             (match line
               [(regexp pattern (list _ value)) (string-trim value)]
               [_ #f]))))))

(define (parse-cpu-times text)
  (define fields
    (match (and (string? text) (string-split text))
      [(cons "cpu" rest) (map string->number rest)]
      [_ '()]))
  (match fields
    [(list user nice system idle iowait irq softirq steal _ ...)
     #:when (andmap exact-nonnegative-integer? fields)
     (define busy (+ user nice system irq softirq steal))
     (cons busy (+ busy idle iowait))]
    [_ #f]))

(define (busy-cores #:interval [interval 1.0])
  (define (sample) (parse-cpu-times (read-first-line "/proc/stat")))
  (define before (sample))
  (cond
    [(not before) #f]
    [else
     (sleep interval)
     (match-define (cons busy total) (sample))
     (define elapsed (- total (cdr before)))
     (if (positive? elapsed)
         (* (processor-count) (/ (- busy (car before)) elapsed 1.0))
         0.0)]))

(define-syntax-parse-rule (with-temporary-path (name:id make:expr) body:expr ...+)
  (let ([name make])
    (dynamic-wind
     void
     (lambda () body ...)
     (lambda () (delete-directory/files name #:must-exist? #f)))))

(define (thread-count text)
  (define n (and text (string->number text)))
  (and (exact-positive-integer? n) n))

(define (thread-setting #:aten aten #:omp omp #:mkl mkl #:cores cores)
  (define mkl-n (thread-count mkl))
  (define omp-n (thread-count omp))
  (define-values (threads source)
    (cond
      [(exact-positive-integer? aten) (values aten "at::get_num_threads")]
      [mkl-n (values mkl-n "MKL_NUM_THREADS")]
      [omp-n (values omp-n "OMP_NUM_THREADS")]
      [else (values cores "default")]))
  (hasheq 'threads threads 'threads_source source
          'omp_num_threads omp 'mkl_num_threads mkl))

(struct gpu (index uuid name))

(define (listing->gpus text)
  (filter-map
   (lambda (line)
     (match (regexp-match #px"^\\s*(\\d+)\\s*,\\s*(\\S+)\\s*,\\s*(.*\\S)\\s*$" line)
       [(list _ index uuid name) (gpu (string->number index) uuid name)]
       [#f #f]))
   (string-split (or text "") "\n")))

(define (cuda-gpu-name visible order listing)
  (define gpus (listing->gpus listing))
  (define id
    (match (string-split (or visible "") ",")
      [(cons id _) (string-trim id)]
      ['() "0"]))
  (define index (and (regexp-match? #px"^\\d+$" id) (string->number id)))
  (define homogeneous? (= 1 (~> gpus (map gpu-name _) remove-duplicates length)))
  (define candidates
    (cond
      [(regexp-match? #rx"^(GPU|MIG)-" id)
       (filter (lambda (g) (string-prefix? (gpu-uuid g) id)) gpus)]
      [(and index (or (equal? order "PCI_BUS_ID") homogeneous?))
       (filter (lambda (g) (= (gpu-index g) index)) gpus)]
      [else '()]))
  (match candidates
    [(list g) (gpu-name g)]
    [_ #f]))

(define (host-meta #:device device #:libtorch libtorch #:threads [aten #f]
                   #:repo [repo (current-directory)])
  (define (git . args) (apply command-output "git" "-C" (path->string repo) args))
  (define sha (git "rev-parse" "HEAD"))
  (define status (git "status" "--porcelain" "--untracked-files=no"))
  (define threads (thread-setting #:aten aten #:omp (getenv "OMP_NUM_THREADS")
                                  #:mkl (getenv "MKL_NUM_THREADS") #:cores (processor-count)))
  (hasheq 'git_sha (or sha "unknown")
          'git_dirty (and status (positive? (string-length status)))
          'racket (string-append (version) " " (symbol->string (system-type 'vm)))
          'libtorch libtorch
          'device (~a device)
          'gpu (and (eq? device 'cuda)
                    (cuda-gpu-name (getenv "CUDA_VISIBLE_DEVICES")
                                   (getenv "CUDA_DEVICE_ORDER")
                                   (command-output "nvidia-smi"
                                                   "--query-gpu=index,uuid,name"
                                                   "--format=csv,noheader")))
          'threads (hash-ref threads 'threads)
          'threads_source (hash-ref threads 'threads_source)
          'omp_num_threads (hash-ref threads 'omp_num_threads)
          'mkl_num_threads (hash-ref threads 'mkl_num_threads)
          'cores (processor-count)
          'cpu (or (proc-field "/proc/cpuinfo" "model name")
                   (command-output "sysctl" "-n" "machdep.cpu.brand_string"))
          'affinity (proc-field "/proc/self/status" "Cpus_allowed_list")
          'host (command-output "hostname")
          'os (~a (system-type 'os*) "-" (system-type 'arch))))

(define (result->record r #:suite suite #:group group #:unit unit
                        #:meta meta #:scale [scale 1e6] #:extra [extra (hasheq)])
  (define per-call (map (lambda (ms) (* ms scale)) (result-per-call r)))
  (for/fold ([h (hasheq 'schema 1
                        'suite suite
                        'group group
                        'case (~a (result-name r))
                        'unit unit
                        'reps (result-reps r)
                        'samples per-call
                        'gc_ms (result-gc-ms r)
                        'load (car (result-load r))
                        'load_end (cadr (result-load r))
                        'stats (summarize per-call)
                        'meta meta)])
            ([(k v) (in-hash extra)])
    (hash-set h k v)))

(define (write-record record [out (current-output-port)])
  (write-string (jsexpr->string record) out)
  (newline out)
  (flush-output out))

(define (format-result r #:unit unit #:scale [scale 1e6])
  (define s (~> r result-per-call (map (lambda (ms) (* ms scale)) _) summarize))
  (format "~a ~a ~a  IQR ~a  gc ~a ms"
          (~a (result-name r) #:min-width 28)
          (~> s (hash-ref 'median) (~r #:precision '(= 1)) (~a #:min-width 10 #:align 'right))
          unit
          (~r (hash-ref s 'iqr) #:precision '(= 1))
          (apply + (result-gc-ms r))))

(module+ test
  (require (only-in json string->jsexpr)
           (only-in racket/file make-temporary-directory make-temporary-file)
           (only-in racket/list make-list)
           rackunit)

  (test-case "quantiles interpolate between order statistics"
    (check-= (quantile '(1 2 3 4) 0.5) 2.5 1e-12)
    (check-= (quantile '(4 1 3 2) 0.25) 1.75 1e-12)
    (check-equal? (quantile '(7) 0.75) 7)
    (define s (summarize '(1 2 3 4 5)))
    (check-equal? (hash-ref s 'median) 3.0)
    (check-equal? (hash-ref s 'iqr) 2.0)
    (check-equal? (hash-ref s 'n) 5))

  (test-case "a list's mean and median skip what is not there"
    (check-equal? (mean '(1 2 6)) 3.0)
    (check-false (mean '()))
    (check-equal? (list-median '(5 1 3)) 3.0)
    (check-equal? (list-median '(10.0 12.0 x)) 11.0)
    (check-false (list-median '()))
    (check-false (list-median #f)))

  (test-case "rounds rotate the case order, so two variants alternate"
    (check-equal? (rotate '(a b c) 1) '(b c a))
    (check-equal? (rotate '(a b) 3) '(b a))
    (check-equal? (rotate '() 2) '())
    (define order '())
    (define (noting name)
      (bench-case name (lambda (_reps) (set! order (cons name order)))))
    (define results
      (measure (list (noting 'a) (noting 'b)) #:warmup 0 #:rounds 4
               #:load (lambda () '(1.0 2.0 3.0))))
    (check-equal? (reverse order) '(a b b a a b b a))
    (check-equal? (map result-name results) '(a b))
    (check-equal? (length (result-per-call (car results))) 4)
    (check-equal? (result-load (car results)) '((1.0 2.0 3.0) (1.0 2.0 3.0))))

  (test-case "with on-failure, a failed case keeps its earlier samples and the others run on"
    (define calls 0)
    (define (flaky _reps)
      (set! calls (add1 calls))
      (when (= calls 2)
        (raise-user-error 'flaky "round two")))
    (define failures '())
    (define (failed! e) (set! failures (cons (exn-message e) failures)))
    (define results
      (measure (list (bench-case 'a void) (bench-case 'b flaky) (bench-case 'c void))
               #:warmup 0 #:rounds 3 #:load void #:on-failure failed!))
    (check-equal? failures '("flaky: round two"))
    (check-equal? calls 2 "a failed case is not run again")
    (check-equal? (map (lambda (r) (length (result-per-call r))) results) '(3 1 3))
    (check-exn #rx"round two"
               (lambda ()
                 (set! calls 1)
                 (measure (list (bench-case 'b flaky)) #:warmup 0 #:rounds 1 #:load void))))

  (test-case "a case may time itself"
    (define r
      (car (measure (list (bench-case 'self (lambda (reps) (* 10.0 reps))
                                      #:reps 4 #:self-timed? #t))
                    #:warmup 2 #:rounds 3 #:load void)))
    (check-equal? (result-per-call r) '(10.0 10.0 10.0))
    (check-equal? (result-reps r) 4))

  (test-case "before runs every round, outside the timed span"
    (define befores 0)
    (define r
      (car (measure (list (bench-case 'quick void
                                      #:before (lambda ()
                                                 (set! befores (add1 befores))
                                                 (sleep 0.05))))
                    #:warmup 1 #:rounds 2 #:load void)))
    (check-equal? befores 3)
    (for ([ms (in-list (result-per-call r))])
      (check-true (< ms 50.0) (format "~a ms" ms))))

  (test-case "busy cores come from the aggregate cpu line"
    (check-equal? (parse-cpu-times "cpu  100 5 20 800 30 1 2 3 7 0") '(131 . 961))
    (check-false (parse-cpu-times "cpu0 100 5 20 800 30 1 2 3 7 0"))
    (check-false (parse-cpu-times "cpu 1 2 3"))
    (check-false (parse-cpu-times #f))
    (define busy (busy-cores #:interval 0.05))
    (check-true (or (not busy) (<= 0.0 busy (processor-count)))))

  (test-case "a temporary path is removed however its body leaves"
    (define kept #f)
    (check-exn #rx"boom"
               (lambda ()
                 (with-temporary-path (dir (make-temporary-directory))
                   (set! kept dir)
                   (call-with-output-file (build-path dir "f") void)
                   (error "boom"))))
    (check-false (directory-exists? kept))
    (with-temporary-path (file (make-temporary-file))
      (set! kept file))
    (check-false (file-exists? kept)))

  (test-case "the paired ratio's interval comes from order statistics"
    (define a '(10 10 10 10 10 10 10 10))
    (define b '(11 12 13 14 15 16 17 18))
    (define cmp (paired-ratio a b))
    (check-= (hash-ref cmp 'median) 1.45 1e-12)
    (check-equal? (hash-ref cmp 'lo) 1.1)
    (check-equal? (hash-ref cmp 'hi) 1.8)
    (check-= (hash-ref cmp 'confidence) (- 1 (/ 2 256)) 1e-12)
    (define small (paired-ratio '(1 1 1) '(2 3 4)))
    (check-= (hash-ref small 'confidence) 0.75 1e-12)
    (check-equal? (hash-ref (paired-ratio (make-list 9 1) (build-list 9 add1)) 'lo)
                  2.0))

  (test-case "load averages parse from Linux and Darwin"
    (check-equal? (parse-load-average "1.50 2.25 3.00 4/1701 99") '(1.5 2.25 3.0))
    (check-equal? (parse-load-average "{ 1.23 1.45 1.67 }") '(1.23 1.45 1.67))
    (check-false (parse-load-average "")))

  (test-case "threads come from ATen, else the variable MKL ops obey"
    (define (label aten omp mkl)
      (define t (thread-setting #:aten aten #:omp omp #:mkl mkl #:cores 16))
      (list (hash-ref t 'threads) (hash-ref t 'threads_source)))
    (check-equal? (label 3 "2" "3") '(3 "at::get_num_threads"))
    (check-equal? (label #f "2" "3") '(3 "MKL_NUM_THREADS"))
    (check-equal? (label #f "2" "x") '(2 "OMP_NUM_THREADS"))
    (check-equal? (label #f #f "0") '(16 "default"))
    (define t (thread-setting #:aten 8 #:omp #f #:mkl "3" #:cores 16))
    (check-false (hash-ref t 'omp_num_threads))
    (check-equal? (hash-ref t 'mkl_num_threads) "3")
    (check-equal? (hash-ref (host-meta #:device 'cpu #:libtorch "x" #:threads 5) 'threads_source)
                  "at::get_num_threads"))

  (test-case "the GPU is the selected device's when the listing pins it down"
    (define mixed
      (string-append "0, GPU-8a1f0c2d-0000-0000-0000-000000000000, NVIDIA A100-SXM4-40GB\n"
                     "1, GPU-8a1f77c2-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090\n"
                     "2, GPU-5fd7e1aa-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090\n"))
    (define same
      (string-append "0, GPU-11110000-0000-0000-0000-000000000000, NVIDIA L4\n"
                     "1, GPU-22220000-0000-0000-0000-000000000000, NVIDIA L4\n"))
    (check-equal? (cuda-gpu-name "GPU-5fd7,GPU-8a1f0" #f mixed) "NVIDIA GeForce RTX 3090")
    (check-equal? (cuda-gpu-name "GPU-8a1f0c" #f mixed) "NVIDIA A100-SXM4-40GB")
    (check-false (cuda-gpu-name "GPU-8a1f" "PCI_BUS_ID" mixed))
    (check-false (cuda-gpu-name "GPU-ffff" #f mixed))
    (check-false (cuda-gpu-name "MIG-8a1f0c2d" #f mixed))
    (check-equal? (cuda-gpu-name "1,2" "PCI_BUS_ID" mixed) "NVIDIA GeForce RTX 3090")
    (check-equal? (cuda-gpu-name "0" "PCI_BUS_ID" mixed) "NVIDIA A100-SXM4-40GB")
    (check-equal? (cuda-gpu-name "1" #f same) "NVIDIA L4")
    (check-false (cuda-gpu-name "1" #f mixed))
    (check-false (cuda-gpu-name "1" "FASTEST_FIRST" mixed))
    (check-false (cuda-gpu-name "7" "PCI_BUS_ID" mixed))
    (check-equal? (cuda-gpu-name #f #f same) "NVIDIA L4")
    (check-equal? (cuda-gpu-name "" #f same) "NVIDIA L4")
    (check-equal? (cuda-gpu-name #f "PCI_BUS_ID" mixed) "NVIDIA A100-SXM4-40GB")
    (check-false (cuda-gpu-name #f #f mixed))
    (check-false (cuda-gpu-name "" #f mixed))
    (check-false (cuda-gpu-name "0" #f #f))
    (check-false (cuda-gpu-name "0" "PCI_BUS_ID" "")))

  (test-case "a record is one JSON line carrying its samples and host"
    (define meta (host-meta #:device 'cpu #:libtorch "2.14.0"))
    (check-equal? (hash-ref meta 'device) "cpu")
    (check-false (hash-ref meta 'gpu))
    (check-true (exact-positive-integer? (hash-ref meta 'threads)))
    (check-equal? (hash-ref (host-meta #:device 'cuda #:libtorch "x") 'device) "cuda")
    (define r (result 'add 100 '(0.001 0.002 0.003) '(0 1 0) '((1.0 1.0 1.0) #f)))
    (define line
      (with-output-to-string
        (lambda ()
          (write-record (result->record r #:suite "micro" #:group "ops" #:unit "ns"
                                        #:meta meta #:extra (hasheq 'device "cpu"))))))
    (define back (string->jsexpr line))
    (check-equal? (hash-ref back 'samples) '(1000.0 2000.0 3000.0))
    (check-equal? (hash-ref (hash-ref back 'stats) 'median) 2000.0)
    (check-equal? (hash-ref back 'case) "add")
    (check-equal? (hash-ref back 'device) "cpu")
    (check-false (hash-ref back 'load_end))
    (check-regexp-match #rx"^add +2000.0 ns  IQR 1000.0  gc 1 ms$"
                        (format-result r #:unit "ns"))))
