#lang racket/base

(require (only-in racket/format ~a)
         (only-in racket/list check-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/string string-split)
         (only-in torch accelerator-if-available device-type torch-version)
         (only-in "contracts.rkt" contract-workloads pipeline-case)
         (only-in "crossings.rkt" crossing-cases)
         (only-in "e2e.rkt"
                  cache-env e2e-cases example-epochs-var example-name example-runner
                  examples find-examples merged-settings settings->jsexpr)
         (only-in "harness.rkt"
                  busy-cores format-result host-meta load-average measure result->record
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
          (match kv
            [(regexp #rx"^([^=]+)=(.*)$" (list _ var value)) (cons var value)]
            [_ (raise-user-error 'run "a variant setting is VAR=VALUE, got ~s" kv)]))))

(define (current-load)
  (or (busy-cores)
      (match (load-average)
        [(cons one-minute _) one-minute]
        [_ #f])))

(define (wait-for-load limit log)
  (let loop ()
    (define load (and limit (current-load)))
    (cond
      [(or (not load) (< load limit)) (void)]
      [else
       (fprintf log "load ~a, waiting for < ~a\n" load limit)
       (sleep 10)
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
  (define twice (check-duplicates (map car variants)))
  (when twice
    (raise-user-error 'run "two variants are named ~a" twice))
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
                 #:max-load max-load #:libtorch libtorch #:out out #:log log
                 #:runner-for [runner-for example-runner])
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
                            #:log log #:runner-for runner-for)
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
     [("--warmup") n "Untimed rounds before them (default 1; e2e runs none)"
                   (set! warmup (string->number n))]
     [("--scale") x "Multiply micro repetitions by x" (set! scale (string->number x))]
     [("--device") d "Device for the ops suite: cpu or cuda" (set! device (string->symbol d))]
     [("--full") "Run examples at their full settings" (set! full? #t)]
     [("--only") names "Comma-separated example numbers" (set! only (string-split names ","))]
     [("--cache") dir "Shared dataset cache root for the examples" (set! cache dir)]
     [("--max-load") l "Wait for a load below l (busy cores over 1 s on Linux)"
                     (set! max-load (string->number l))]
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
  (require (for-syntax racket/base)
           (only-in racket/file make-temporary-file)
           (only-in racket/port open-output-nowhere)
           (only-in racket/runtime-path define-runtime-path)
           rackunit
           (only-in "e2e.rkt" example)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "summary.rkt" read-records summary-markdown))

  (define-runtime-path stepper "fixtures/stepper.rkt")
  (define quiet (open-output-nowhere))

  (define (records-of thunk)
    (with-temporary-path (path (make-temporary-file "bench-~a.jsonl"))
      (call-with-output-file path #:exists 'truncate thunk)
      (read-records (list path))))

  (test-case "a variant names its settings"
    (check-equal? (parse-variant "base") '("base"))
    (check-equal? (parse-variant "one:OMP_NUM_THREADS=1,X=a=b")
                  '("one" ("OMP_NUM_THREADS" . "1") ("X" . "a=b")))
    (check-equal? (parse-variant "alt:LD_LIBRARY_PATH=/a:/b")
                  '("alt" ("LD_LIBRARY_PATH" . "/a:/b")))
    (check-exn #rx"VAR=VALUE" (lambda () (parse-variant "bad:X"))))

  (test-case "every micro suite runs, records and summarises"
    (define records
      (records-of (lambda (out)
                    (run-suites micro-suites #:rounds 1 #:warmup 0 #:scale 0.001
                                #:out out #:log quiet))))
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
               (lambda () (run-suites '(nope) #:log quiet)))
    (check-exn #rx"two variants are named a"
               (lambda () (run-suites '(e2e) #:variants '(("a") ("a" ("X" . "1")))
                                      #:log quiet))))

  (test-case "an e2e run records every child, a summary per variant, and a failure"
    (define fixture (example "stepper" '(("EPOCHS" . 3)) '(("EPOCHS" . 4)) "EPOCHS" 0 'first))
    (define (run variants rounds out)
      (run-e2e (list fixture) #:rounds rounds #:full? #f #:variants variants #:cache #f
               #:max-load #f #:libtorch "x" #:out out #:log quiet
               #:runner-for (lambda (_ex) stepper)))
    (define records
      (records-of (lambda (out)
                    (run '(("a") ("b" ("EPOCHS" . "2"))) 2 out)
                    (run '((#f ("EPOCHS" . "1"))) 1 out))))
    (define (group g) (filter (lambda (r) (equal? (hash-ref r 'group) g)) records))
    (check-equal? (map (lambda (r) (hash-ref r 'steps)) (group "run")) '(9 6 6 9 3))
    (check-regexp-match #rx"no measured steps" (hash-ref (car (group "error")) 'error))
    (define summaries (group "summary"))
    (check-equal? (map (lambda (r) (hash-ref r 'variant)) summaries) '("a" "b"))
    (check-equal? (map (lambda (r) (hash-ref (hash-ref r 'settings) 'EPOCHS)) summaries)
                  '("3" "2"))
    (check-equal? (map (lambda (r) (hash-ref (hash-ref r 'stats) 'n)) summaries) '(2 2))
    (define md (summary-markdown records))
    (check-regexp-match #rx"## End to end" md)
    (check-regexp-match #rx"\\| stepper \\| b vs a \\|" md)
    (check-regexp-match #rx"n < 6" md)))
