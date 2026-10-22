#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file file->lines make-temporary-directory)
           (only-in racket/port port->string)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-contains? string-prefix?)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "harness.rkt" with-temporary-path))

  (define-runtime-path script "l0-baseline.sh")
  (define-runtime-path fake-bin "fixtures/fake-bin")
  (define bash (find-executable-path "bash"))

  (define (run-baseline dir stage #:rounds [rounds #f])
    (define env (environment-variables-copy (current-environment-variables)))
    (define (set-var! name value)
      (environment-variables-set! env (string->bytes/utf-8 name)
                                  (and value (string->bytes/utf-8 value))))
    (set-var! "PATH" (string-append (path->string fake-bin) ":" (or (getenv "PATH") "")))
    (set-var! "FAKE_LOG" (path->string (build-path dir "calls.log")))
    (set-var! "BASELINE_OUT" (path->string (build-path dir "l0")))
    (set-var! "BASELINE_SMOKE" #f)
    (set-var! "BASELINE_ROUNDS" rounds)
    (define-values (sp out in err)
      (parameterize ([current-environment-variables env])
        (subprocess #f #f #f bash (path->string script) stage)))
    (close-output-port in)
    (define drains (for/list ([p (list out err)]) (thread (lambda () (port->string p)))))
    (define status
      (cond
        [(sync/timeout 120 sp) (subprocess-status sp)]
        [else (subprocess-kill sp #t) 'timeout]))
    (for ([t (in-list drains)]) (sync/timeout 10 t))
    (close-input-port out)
    (close-input-port err)
    status)

  (define (bench-calls dir)
    (for/list ([line (in-list (file->lines (build-path dir "calls.log")))]
               #:when (string-prefix? line "racket scripts/bench/run.rkt"))
      line))

  (define (jsonl dir) (file->lines (build-path dir "l0.jsonl")))

  (define (in-fresh-dir proc)
    (with-temporary-path (dir (make-temporary-directory "l0-baseline-~a"))
      (proc dir)))

  (cond
    [(not bash) (displayln "[l0-baseline] skipped: no bash")]
    [else
     (call-with-process-deadline
      600
      (lambda ()
        (test-case "the full run starts the JSONL afresh, and its stages append to it"
          (in-fresh-dir
           (lambda (dir)
             (call-with-output-file (build-path dir "l0.jsonl") (lambda (o) (displayln "stale" o)))
             (check-equal? (run-baseline dir "all") 0)
             (define records (jsonl dir))
             (check-false (member "stale" records))
             (check-equal? (length records) (length (bench-calls dir)))
             (check-true (> (length records) 1)))))

        (test-case "a single stage appends to the JSONL"
          (in-fresh-dir
           (lambda (dir)
             (call-with-output-file (build-path dir "l0.jsonl") (lambda (o) (displayln "kept" o)))
             (check-equal? (run-baseline dir "cpu") 0)
             (check-equal? (car (jsonl dir)) "kept"))))

        (test-case "BASELINE_ROUNDS reaches every suite"
          (in-fresh-dir
           (lambda (dir)
             (for ([stage (in-list '("cpu" "cuda"))])
               (check-equal? (run-baseline dir stage #:rounds "7") 0))
             (define calls (bench-calls dir))
             (check-true (> (length calls) 4))
             (for ([call (in-list calls)])
               (check-true (string-contains? call " --rounds 7 ") call)))))

        (test-case "unset, BASELINE_ROUNDS leaves e2e at 3 and the others at their defaults"
          (in-fresh-dir
           (lambda (dir)
             (for ([stage (in-list '("cpu" "cuda"))])
               (check-equal? (run-baseline dir stage) 0))
             (for ([call (in-list (bench-calls dir))])
               (if (string-contains? call " e2e ")
                   (check-true (string-contains? call " --rounds 3 ") call)
                   (check-false (string-contains? call "--rounds") call))))))))]))
