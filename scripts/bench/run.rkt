#lang racket/base

(require (only-in racket/format ~a)
         (only-in racket/match match-define)
         (only-in racket/string string-split)
         (only-in torch accelerator-if-available device-type torch-version)
         (only-in "contracts.rkt" contract-workloads pipeline-case)
         (only-in "crossings.rkt" crossing-cases)
         (only-in "e2e.rkt"
                  cache-env e2e-cases example-epochs-var example-name examples
                  find-examples merged-settings settings->jsexpr)
         (only-in "harness.rkt"
                  format-result host-meta load-average measure result->record
                  result-name summarize write-record)
         (only-in "layers.rkt" layer-cases)
         (only-in "ops.rkt" op-cases))

(provide run-suites
         parse-variant
         micro-suites)

(define micro-suites '(crossings layers ops contracts))

(define (parse-variant text)
  (match-define (list _ name settings) (regexp-match #rx"^([^:]*)(?::(.*))?$" text))
  (cons name
        (for/list ([kv (in-list (if settings (string-split settings ",") '()))])
          (define m (regexp-match #rx"^([^=]+)=(.*)$" kv))
          (unless m (raise-user-error 'run "a variant setting is VAR=VALUE, got ~s" kv))
          (cons (cadr m) (caddr m)))))

(define (wait-for-load limit log)
  (let loop ()
    (define load (load-average))
    (cond
      [(or (not limit) (not load) (< (car load) limit)) (void)]
      [else
       (fprintf log "load ~a, waiting for < ~a\n" (car load) limit)
       (sleep 30)
       (loop)])))

(define (run-suites suites
                    #:rounds [rounds #f]
                    #:warmup [warmup 1]
                    #:scale [scale 1]
                    #:device [device 'cpu]
                    #:full? [full? #f]
                    #:only [only #f]
                    #:variants [variants '((#f))]
                    #:cache [cache #f]
                    #:max-load [max-load #f]
                    #:out [out (current-output-port)]
                    #:log [log (current-error-port)])
  (define libtorch (torch-version))
  (define (n reps) (max 1 (inexact->exact (round (* reps scale)))))
  (define (measure-group group cases #:unit [unit "ns"] #:scale [unit-scale 1e6]
                         #:case-name [case-name #f] #:on-device [on-device 'cpu])
    (wait-for-load max-load log)
    (define meta (host-meta #:device on-device #:libtorch libtorch))
    (fprintf log "~a~a\n" group (if case-name (format " ~a" case-name) ""))
    (for ([r (in-list (measure cases #:warmup warmup #:rounds (or rounds 5)))])
      (fprintf log "  ~a\n" (format-result r #:unit unit #:scale unit-scale))
      (write-record
       (result->record r #:suite "micro" #:group (~a group) #:unit unit #:meta meta
                       #:scale unit-scale
                       #:extra (if case-name
                                   (hasheq 'case (~a case-name) 'variant (~a (result-name r)))
                                   (hasheq)))
       out)))
  (for ([suite (in-list suites)])
    (case suite
      [(crossings) (measure-group 'crossings (crossing-cases #:reps (n 200000)))]
      [(layers) (measure-group 'layers (layer-cases #:reps (n 20000) #:drain (n 2000)))]
      [(ops)
       (for ([o (in-list (op-cases #:device device #:scale scale))])
         (measure-group 'ops (cdr o) #:case-name (car o) #:on-device device))]
      [(contracts)
       (for ([w (in-list (contract-workloads #:scale scale))])
         (measure-group 'contracts (cdr w) #:case-name (car w)))
       (measure-group 'pipeline (list (pipeline-case)) #:unit "ms" #:scale 1)]
      [(e2e)
       (run-e2e (if only (find-examples only) examples)
                #:rounds (or rounds 1) #:full? full? #:variants variants #:cache cache
                #:max-load max-load #:libtorch libtorch #:out out #:log log)]
      [else (raise-user-error 'run "unknown suite ~a" suite)])))

(define (run-e2e exs #:rounds rounds #:full? full? #:variants variants #:cache cache
                 #:max-load max-load #:libtorch libtorch #:out out #:log log)
  (define device (device-type (accelerator-if-available)))
  (define scale (if full? 'full 'short))
  (define cached (cache-env cache))
  (define with-cache
    (for/list ([v (in-list variants)]) (cons (car v) (append cached (cdr v)))))
  (for ([ex (in-list exs)])
    (wait-for-load max-load log)
    (define meta (host-meta #:device device #:libtorch libtorch))
    (define rates (make-hash))
    (define gcs (make-hash))
    (define metas (make-hash))
    (define (on-run run-of variant analysis)
      (hash-update! rates variant (lambda (xs) (cons (hash-ref analysis 'steps_per_s 0.0) xs)) '())
      (hash-update! gcs variant (lambda (xs) (cons (hash-ref analysis 'gc_ms 0) xs)) '())
      (hash-ref! metas variant (lambda () (or (hash-ref analysis 'meta #f) meta)))
      (fprintf log "~a~a: ~a steps/s, epochs ~a s\n" (example-name run-of)
               (if variant (format "/~a" variant) "")
               (hash-ref analysis 'steps_per_s #f) (hash-ref analysis 'epoch_s '()))
      (write-record (hash-set* analysis 'schema 1 'suite "e2e" 'group "run"
                               'case (example-name run-of) 'variant variant
                               'load (load-average)
                               'meta (or (hash-ref analysis 'meta #f) meta))
                    out))
    (define unit (if (example-epochs-var ex) "s/epoch" "ms/step"))
    (with-handlers ([exn:fail:user?
                     (lambda (e)
                       (fprintf log "~a\n" (exn-message e))
                       (write-record (hasheq 'schema 1 'suite "e2e" 'group "error"
                                             'case (example-name ex) 'error (exn-message e)
                                             'meta meta)
                                     out))])
      (define results
        (measure (e2e-cases (list ex) #:scale scale #:variants with-cache #:on-run on-run
                            #:log log)
                 #:warmup 0 #:rounds rounds))
      (for ([r (in-list results)] [v (in-list variants)])
        (write-record
         (result->record r #:suite "e2e" #:group "summary" #:unit unit
                         #:meta (hash-ref metas (car v) meta)
                         #:scale (if (example-epochs-var ex) 1e-3 1)
                         #:extra (hasheq 'case (example-name ex)
                                         'variant (car v)
                                         'settings (settings->jsexpr
                                                    (merged-settings ex scale (cdr v)))
                                         'gc_ms (reverse (hash-ref gcs (car v)))
                                         'steps_per_s (summarize (hash-ref rates (car v)))))
         out)))))

(module+ main
  (require (only-in racket/cmdline command-line))
  (define rounds #f)
  (define warmup 1)
  (define scale 1)
  (define device 'cpu)
  (define full? #f)
  (define only #f)
  (define variants '())
  (define cache #f)
  (define max-load #f)
  (define out-path #f)
  (define suites
    (command-line
     #:once-each
     [("--rounds") n "Timed rounds (default 5; e2e 1)" (set! rounds (string->number n))]
     [("--warmup") n "Untimed rounds before them (default 1)" (set! warmup (string->number n))]
     [("--scale") x "Multiply micro repetitions by x" (set! scale (string->number x))]
     [("--device") d "Device for the ops suite: cpu or cuda" (set! device (string->symbol d))]
     [("--full") "Run examples at their full settings" (set! full? #t)]
     [("--only") names "Comma-separated example numbers" (set! only (string-split names ","))]
     [("--cache") dir "Shared dataset cache root for the examples" (set! cache dir)]
     [("--max-load") l "Wait for a 1-minute load below l" (set! max-load (string->number l))]
     [("--out") file "Append JSON lines to file" (set! out-path file)]
     #:multi
     [("--variant") v "name:VAR=VALUE,... (repeat for A/B)"
                    (set! variants (append variants (list (parse-variant v))))]
     #:args suite-names
     (for*/list ([s (in-list suite-names)]
                 [x (in-list (if (equal? s "micro") (map symbol->string micro-suites) (list s)))])
       (string->symbol x))))
  (define (go out)
    (run-suites suites
                #:rounds rounds
                #:warmup warmup #:scale scale #:device device #:full? full? #:only only
                #:variants (if (null? variants) '((#f)) variants)
                #:cache cache #:max-load max-load #:out out))
  (if out-path
      (call-with-output-file out-path #:exists 'append go)
      (go (current-output-port))))

(module+ test
  (require (only-in racket/file make-temporary-file)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "summary.rkt" read-records summary-markdown))

  (test-case "a variant names its settings"
    (check-equal? (parse-variant "base") '("base"))
    (check-equal? (parse-variant "one:OMP_NUM_THREADS=1,X=a=b")
                  '("one" ("OMP_NUM_THREADS" . "1") ("X" . "a=b")))
    (check-equal? (parse-variant "alt:LD_LIBRARY_PATH=/a:/b")
                  '("alt" ("LD_LIBRARY_PATH" . "/a:/b")))
    (check-exn #rx"VAR=VALUE" (lambda () (parse-variant "bad:X"))))

  (test-case "every micro suite runs, records and summarises"
    (define path (make-temporary-file "bench-~a.jsonl"))
    (call-with-output-file path #:exists 'truncate
      (lambda (out)
        (run-suites micro-suites #:rounds 1 #:warmup 0 #:scale 0.001
                    #:out out #:log (open-output-nowhere))))
    (define records (read-records (list path)))
    (delete-file path)
    (define groups (for/list ([r (in-list records)]) (hash-ref r 'group)))
    (for ([g (in-list '("crossings" "layers" "ops" "contracts" "pipeline"))])
      (check-not-false (member g groups) g))
    (for ([r (in-list records)] #:unless (equal? (hash-ref r 'case) "finalization"))
      (check-true (positive? (hash-ref (hash-ref r 'stats) 'median)) (hash-ref r 'case)))
    (define md (summary-markdown records))
    (for ([heading (in-list '("## Native crossings" "## The op stack" "## Per op"
                              "## Validation strategies" "## A real pipeline"))])
      (check-regexp-match (regexp-quote heading) md))
    (check-exn #rx"unknown suite"
               (lambda () (run-suites '(nope) #:log (open-output-nowhere))))))
