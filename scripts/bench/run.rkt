#lang racket/base

(require (only-in racket/cmdline command-line)
         (only-in racket/format ~a)
         (only-in racket/list check-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/string string-split)
         (only-in torch torch-version)
         (only-in "contracts.rkt" contract-workloads getter-cases pipeline-case)
         (only-in "crossings.rkt" crossing-cases)
         (only-in "e2e-suite.rkt" run-e2e)
         (only-in "e2e.rkt" examples find-examples)
         (only-in "harness.rkt" format-result host-meta measure result->record result-name)
         (only-in "l0.rkt" l0-suites run-l0-suite)
         (only-in "layers.rkt" layer-cases)
         (only-in "ops.rkt" op-cases)
         (only-in "session.rkt" fresh-run-id record-writer wait-for-load)
         (only-in "shim.rkt" aten-threads))

(provide run-suites
         parse-arguments
         parse-variant
         micro-suites)

(define micro-suites '(crossings layers ops contracts))
(define known-suites (append micro-suites '(e2e) l0-suites))

(define (check-suites suites)
  (define unknown (for/first ([s (in-list suites)] #:unless (memq s known-suites)) s))
  (when unknown
    (raise-user-error 'run "unknown suite ~a; known: micro ~a" unknown known-suites)))

(define (parse-variant text)
  (match-define (list _ name settings) (regexp-match #rx"^([^:]*)(?::(.*))?$" text))
  (cons name
        (for/list ([kv (in-list (if settings (string-split settings ",") '()))])
          (match kv
            [(regexp #rx"^([^=]+)=(.*)$" (list _ var value)) (cons var value)]
            [_ (raise-user-error 'run "a variant setting is VAR=VALUE, got ~s" kv)]))))

(define (flag-value flag parse ok? expected text)
  (match (parse text)
    [(? ok? v) v]
    [_ (raise-user-error 'run "~a expects ~a, got ~s" flag expected text)]))

(define (finite-positive-real? x)
  (and (real? x) (< 0 x +inf.0)))

(define (ops-device? d)
  (and (memq d '(cpu cuda mps)) #t))

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
                    #:time-limit [time-limit #f]
                    #:run-id [run-id (fresh-run-id)]
                    #:out [out (current-output-port)]
                    #:log [log (current-error-port)])
  (check-suites suites)
  (define libtorch (torch-version))
  (define emit (record-writer run-id out))
  (define (n reps) (max 1 (inexact->exact (round (* reps scale)))))
  (define twice (check-duplicates (map car variants)))
  (when twice
    (raise-user-error 'run "two variants are named ~a" twice))
  (define (measure-group group cases #:unit [unit "ns"] #:scale [unit-scale 1e6]
                         #:case-name [case-name #f] #:on-device [on-device 'cpu])
    (wait-for-load max-load log)
    (define meta (host-meta #:device on-device #:libtorch libtorch #:threads (aten-threads)))
    (fprintf log "~a~a\n" group (if case-name (format " ~a" case-name) ""))
    (for ([r (in-list (measure cases #:warmup warmup #:rounds (or rounds 5)))])
      (fprintf log "  ~a\n" (format-result r #:unit unit #:scale unit-scale))
      (emit
       (result->record r #:suite "micro" #:group (~a group) #:unit unit #:meta meta
                       #:scale unit-scale
                       #:extra (if case-name
                                   (hasheq 'impl "rktorch" 'case (~a case-name)
                                           'variant (~a (result-name r)))
                                   (hasheq 'impl "rktorch"))))))
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
       (measure-group 'getters (getter-cases #:reps (n 200000)))
       (measure-group 'pipeline (list (pipeline-case #:reps (n 1))) #:unit "ms" #:scale 1)]
      [(pytorch-ops threads pytorch-threads calls host-gpu)
       (wait-for-load max-load log)
       (fprintf log "~a\n" suite)
       (run-l0-suite suite #:emit emit #:log log #:device device #:scale scale
                     #:rounds rounds #:warmup warmup #:libtorch libtorch)]
      [(e2e)
       (run-e2e (if only (find-examples only) examples)
                #:rounds (or rounds 1) #:full? full? #:variants variants #:cache cache
                #:max-load max-load #:time-limit time-limit #:libtorch libtorch #:run-id run-id
                #:out out #:log log)])))

(define (parse-arguments argv)
  (define options (make-hasheq))
  (define (option! key value) (hash-set! options key value))
  (define names
    (let parse ([argv argv] [names '()])
      (command-line
       #:program "run.rkt"
       #:argv argv
       #:usage-help "Suites: micro crossings layers ops contracts e2e,"
                    "and pytorch-ops threads pytorch-threads calls host-gpu."
                    "Options go before or after the suites."
       #:once-each
       [("--rounds") n "Timed rounds (default 5; e2e 1)"
                     (option! 'rounds
                              (flag-value "--rounds" string->number exact-positive-integer?
                                          "a positive integer" n))]
       [("--warmup") n "Untimed rounds before them (default 1; e2e runs none)"
                     (option! 'warmup
                              (flag-value "--warmup" string->number exact-nonnegative-integer?
                                          "a non-negative integer" n))]
       [("--scale") x "Multiply micro repetitions by x"
                    (option! 'scale
                             (flag-value "--scale" string->number finite-positive-real?
                                         "a finite positive number" x))]
       [("--device") d "Device for ops, pytorch-ops, calls and host-gpu: cpu, cuda or mps"
                     (option! 'device
                              (flag-value "--device" string->symbol ops-device? "cpu, cuda or mps"
                                          d))]
       [("--full") "Run examples at their full settings" (option! 'full? #t)]
       [("--only") names "Comma-separated example numbers"
                   (option! 'only (string-split names ","))]
       [("--cache") dir "Shared dataset cache root for the examples" (option! 'cache dir)]
       [("--max-load") l "Wait for a load below l (busy cores over 1 s on Linux)"
                       (option! 'max-load
                                (flag-value "--max-load" string->number finite-positive-real?
                                            "a finite positive number" l))]
       [("--time-limit") s "Kill an e2e child after s seconds (default 3600; --full 43200)"
                         (option! 'time-limit
                                  (flag-value "--time-limit" string->number finite-positive-real?
                                              "a finite positive number" s))]
       [("--out") file "Append JSON lines to file" (option! 'out file)]
       #:multi
       [("--variant") v "name:VAR=VALUE,... (repeat for A/B)"
                      (hash-update! options 'variants
                                    (lambda (vs) (append vs (list (parse-variant v))))
                                    '())]
       #:args suite
       (match suite
         ['() (reverse names)]
         [(cons name more) (parse more (cons name names))]))))
  (define suites
    (for*/list ([s (in-list names)]
                [x (in-list (if (equal? s "micro") (map symbol->string micro-suites) (list s)))])
      (string->symbol x)))
  (check-suites suites)
  (values suites options))

(module+ main
  (define-values (suites options)
    (parse-arguments (current-command-line-arguments)))
  (define (option key [default #f]) (hash-ref options key default))
  (define (go out)
    (run-suites suites
                #:rounds (option 'rounds)
                #:warmup (option 'warmup 1) #:scale (option 'scale 1)
                #:device (option 'device 'cpu) #:full? (option 'full?) #:only (option 'only)
                #:variants (option 'variants '((#f)))
                #:cache (option 'cache) #:max-load (option 'max-load)
                #:time-limit (option 'time-limit) #:out out))
  (match (option 'out)
    [#f (go (current-output-port))]
    [path (call-with-output-file path #:exists 'append go)]))

(module+ test
  (require (only-in racket/file make-temporary-file)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "harness.rkt" with-temporary-path)
           (only-in "summary.rkt" read-records summary-markdown))

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
    (for ([g (in-list '("crossings" "layers" "ops" "contracts" "getters" "pipeline"))])
      (check-not-false (member g groups) g))
    (for ([r (in-list records)] #:unless (equal? (hash-ref r 'case) "finalization"))
      (check-true (positive? (hash-ref (hash-ref r 'stats) 'median)) (hash-ref r 'case)))
    (define md (summary-markdown records))
    (for ([heading (in-list '("## Native crossings" "## The op stack" "## Per op"
                              "## Validation strategies" "## Short and long getters"
                              "## A real pipeline"))])
      (check-regexp-match (regexp-quote heading) md))
    (check-exn #rx"unknown suite nope"
               (lambda () (run-suites '(nope) #:log quiet)))
    (check-exn #rx"two variants are named a"
               (lambda () (run-suites '(e2e) #:variants '(("a") ("a" ("X" . "1")))
                                      #:log quiet))))

  (test-case "flags go before or after the suites"
    (define (parsed . argv)
      (define-values (suites options) (parse-arguments (list->vector argv)))
      (list suites options))
    (check-equal? (parsed "micro" "--out" "r.jsonl" "--max-load" "3")
                  (list micro-suites (make-hasheq '((out . "r.jsonl") (max-load . 3)))))
    (check-equal? (parsed "--max-load" "3" "ops" "--device" "cuda" "--out" "r.jsonl")
                  (list '(ops) (make-hasheq '((max-load . 3) (device . cuda) (out . "r.jsonl")))))
    (check-equal? (parsed "e2e" "--cache" "c" "--variant" "a" "--out" "r.jsonl"
                          "--variant" "b:X=1")
                  (list '(e2e) (make-hasheq '((cache . "c") (out . "r.jsonl")
                                              (variants . (("a") ("b" ("X" . "1"))))))))
    (check-equal? (parsed "crossings" "--rounds" "2" "layers")
                  (list '(crossings layers) (make-hasheq '((rounds . 2)))))
    (check-equal? (parsed "e2e" "--time-limit" "5")
                  (list '(e2e) (make-hasheq '((time-limit . 5))))))

  (test-case "a flag rejects what is not its value"
    (define (fails flag value expected)
      (check-exn (regexp (string-append (regexp-quote flag) " expects " expected ", got \""
                                        (regexp-quote value) "\""))
                 (lambda () (parse-arguments (vector "micro" flag value)))))
    (fails "--rounds" "six" "a positive integer")
    (fails "--rounds" "0" "a positive integer")
    (fails "--rounds" "1.5" "a positive integer")
    (fails "--warmup" "-1" "a non-negative integer")
    (fails "--warmup" "one" "a non-negative integer")
    (for* ([flag (in-list '("--scale" "--max-load" "--time-limit"))]
           [value (in-list '("big" "0" "-1" "1+2i" "+inf.0" "-inf.0" "+nan.0"))])
      (fails flag value "a finite positive number"))
    (fails "--device" "gpu" "cpu, cuda or mps")
    (fails "--device" "CUDA" "cpu, cuda or mps")
    (define-values (_suites options)
      (parse-arguments (vector "micro" "--warmup" "0" "--scale" "0.5" "--max-load" "2.5")))
    (check-equal? options (make-hasheq '((warmup . 0) (scale . 0.5) (max-load . 2.5))))
    (for ([d (in-list '("cpu" "cuda" "mps"))])
      (define-values (_ops parsed) (parse-arguments (vector "ops" "--device" d)))
      (check-equal? (hash-ref parsed 'device) (string->symbol d))))

  (test-case "an unknown suite fails before any suite runs"
    (check-exn #rx"unknown suite --out"
               (lambda () (parse-arguments (vector "micro" "--" "--out" "r.jsonl"))))
    (check-exn #rx"unknown suite nope"
               (lambda () (parse-arguments (vector "micro" "nope" "--out" "r.jsonl"))))
    (check-equal? (records-of (lambda (out)
                                (check-exn #rx"unknown suite nope"
                                           (lambda ()
                                             (run-suites '(crossings nope) #:rounds 1 #:warmup 0
                                                         #:scale 0.001 #:out out #:log quiet)))))
                  '())))
