#lang racket/base

(require (only-in json jsexpr->string)
         (only-in racket/format ~a ~r)
         (only-in racket/future processor-count)
         (only-in racket/list drop take)
         (only-in racket/match match-define)
         (only-in racket/port open-output-nowhere port->string with-output-to-string)
         (only-in racket/string string-split string-trim)
         (only-in racket/system process*/ports))

(provide bench-case
         bench-case?
         bench-case-name
         (struct-out result)
         measure
         quantile
         summarize
         paired-ratio
         rotate
         parse-load-average
         load-average
         host-meta
         result->record
         write-record
         format-result)

(struct bench-case (name run reps before self-timed?)
  #:constructor-name make-bench-case
  #:name bench-case-type) ;; noqa

(define (bench-case name run
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
                 #:load [load load-average])
  (for* ([_ (in-range warmup)] [c (in-list cases)])
    (run-once c))
  (define samples (make-hasheq))
  (define start-load (load))
  (for* ([i (in-range rounds)] [c (in-list (rotate cases i))])
    (define sample (run-once c))
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
  (and (file-exists? path)
       (call-with-input-file path
         (lambda (in)
           (for/first ([line (in-lines in)]
                       #:when (regexp-match? (regexp (string-append "^" key)) line))
             (string-trim (cadr (regexp-match #rx":(.*)$" line))))))))

(define (thread-setting)
  (define omp (getenv "OMP_NUM_THREADS"))
  (define n (and omp (string->number omp)))
  (if (exact-positive-integer? n)
      (hasheq 'threads n 'threads_source "OMP_NUM_THREADS")
      (hasheq 'threads (processor-count) 'threads_source "default")))

(define (host-meta #:device device #:libtorch libtorch #:repo [repo (current-directory)])
  (define (git . args) (apply command-output "git" "-C" (path->string repo) args))
  (define sha (git "rev-parse" "HEAD"))
  (define status (git "status" "--porcelain" "--untracked-files=no"))
  (define threads (thread-setting))
  (hasheq 'git_sha (or sha "unknown")
          'git_dirty (and status (positive? (string-length status)))
          'racket (string-append (version) " " (symbol->string (system-type 'vm)))
          'libtorch libtorch
          'device (~a device)
          'gpu (and (eq? device 'cuda)
                    (command-output "nvidia-smi" "--query-gpu=name"
                                    "--format=csv,noheader"))
          'threads (hash-ref threads 'threads)
          'threads_source (hash-ref threads 'threads_source)
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
  (define s (summarize (map (lambda (ms) (* ms scale)) (result-per-call r))))
  (format "~a ~a ~a  IQR ~a  gc ~a ms"
          (~a (result-name r) #:min-width 28)
          (~a (~r (hash-ref s 'median) #:precision '(= 1)) #:min-width 10 #:align 'right)
          unit
          (~r (hash-ref s 'iqr) #:precision '(= 1))
          (apply + (result-gc-ms r))))

(module+ test
  (require (only-in json string->jsexpr)
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

  (test-case "a case may time itself, and before runs untimed"
    (define befores 0)
    (define r
      (car (measure (list (bench-case 'self (lambda (reps) (* 10.0 reps))
                                      #:reps 4 #:self-timed? #t
                                      #:before (lambda () (set! befores (add1 befores)))))
                    #:warmup 2 #:rounds 3 #:load void)))
    (check-equal? (result-per-call r) '(10.0 10.0 10.0))
    (check-equal? befores 5)
    (check-equal? (result-reps r) 4))

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
