#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file file->lines make-temporary-directory)
           (only-in racket/match match-define)
           (only-in racket/port port->string)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-contains? string-prefix?)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "harness.rkt" with-temporary-path))

  (define-runtime-path script "l0-baseline.sh")
  (define-runtime-path fake-bin "fixtures/fake-bin")
  (define bash (find-executable-path "bash"))

  (define (run-baseline dir stage #:rounds [rounds #f] #:out [stem "l0"] #:smoke [smoke #f]
                        #:folder-fails? [folder-fails? #f])
    (define env (environment-variables-copy (current-environment-variables)))
    (define (set-var! name value)
      (environment-variables-set! env (string->bytes/utf-8 name)
                                  (and value (string->bytes/utf-8 value))))
    (set-var! "PATH" (string-append (path->string fake-bin) ":" (or (getenv "PATH") "")))
    (set-var! "FAKE_LOG" (path->string (build-path dir "calls.log")))
    (set-var! "BASELINE_OUT" (and stem (path->string (build-path dir stem))))
    (set-var! "BASELINE_DIR" (path->string (build-path dir "data")))
    (set-var! "BASELINE_DATE" "2026-01-02")
    (set-var! "BASELINE_SMOKE" (and smoke "1"))
    (set-var! "BASELINE_ROUNDS" rounds)
    (set-var! "FAKE_FOLDER_FAILS" (and folder-fails? "1"))
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

  (define (calls-of dir prefix)
    (for/list ([line (in-list (file->lines (build-path dir "calls.log")))]
               #:when (string-prefix? line prefix))
      line))

  (define (bench-calls dir) (calls-of dir "racket scripts/bench/run.rkt"))

  (define (variants-of call)
    (regexp-match* #px"--variant (\\S+)" call #:match-select cadr))

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

        (test-case "unset, BASELINE_OUT puts the records in BASELINE_DIR, outside the repository"
          (in-fresh-dir
           (lambda (dir)
             (check-equal? (run-baseline dir "cpu" #:out #f) 0)
             (check-true (file-exists? (build-path dir "data" "baseline-2026-01-02.jsonl")))
             (check-equal? (run-baseline dir "cpu" #:out #f #:smoke #t) 0)
             (check-true (file-exists? (build-path dir "data" "smoke-2026-01-02.jsonl"))))))

        (test-case "BASELINE_ROUNDS reaches every suite"
          (in-fresh-dir
           (lambda (dir)
             (for ([stage (in-list '("cpu" "cuda" "loader"))])
               (check-equal? (run-baseline dir stage #:rounds "7") 0))
             (define calls (bench-calls dir))
             (check-true (> (length calls) 4))
             (for ([call (in-list calls)])
               (check-true (string-contains? call " --rounds 7 ") call)))))

        (test-case "unset, BASELINE_ROUNDS leaves e2e at 3 and the others at their defaults"
          (in-fresh-dir
           (lambda (dir)
             (for ([stage (in-list '("cpu" "cuda" "loader"))])
               (check-equal? (run-baseline dir stage) 0))
             (for ([call (in-list (bench-calls dir))])
               (if (string-contains? call " e2e ")
                   (check-true (string-contains? call " --rounds 3 ") call)
                   (check-false (string-contains? call "--rounds") call))))))

        (test-case "the loader stage writes the PNG tree, then runs each pipeline and worker count beside its twin"
          (in-fresh-dir
           (lambda (dir)
             (check-equal? (run-baseline dir "loader") 0)
             (check-equal? (calls-of dir "racket scripts/bench/cifar-folder.rkt")
                           '("racket scripts/bench/cifar-folder.rkt full"))
             (match-define (list loader resnet) (bench-calls dir))
             (for ([call (in-list (list loader resnet))])
               (check-true (string-contains? call " --device cuda ") call))
             (check-true (string-contains? loader " --only cifar-loader,py-cifar-loader ") loader)
             (check-true (string-contains? resnet " --only cifar-resnet,py-cifar-resnet ") resnet)
             (define (expected workers)
               (for*/list ([p (in-list '("per-sample" "batched"))] [w (in-list workers)])
                 (format "~a-~a:PIPELINE=~a,WORKERS=~a,PNG_DIGEST=0123456789ab" p w p w)))
             (check-equal? (variants-of loader) (expected '(0 1 2 4 8)))
             (check-equal? (variants-of resnet) (expected '(0 2 4))))))

        (test-case "a smoke run of the loader stage takes the fixture tree and the smallest settings"
          (in-fresh-dir
           (lambda (dir)
             (check-equal? (run-baseline dir "loader" #:smoke #t) 0)
             (check-equal? (calls-of dir "racket scripts/bench/cifar-folder.rkt")
                           '("racket scripts/bench/cifar-folder.rkt fixture"))
             (for ([call (in-list (bench-calls dir))])
               (check-true (string-contains? call " --device cuda ") call))
             (match-define (list loader resnet) (map variants-of (bench-calls dir)))
             (check-equal? (length loader) 4)
             (check-equal? (length resnet) 4)
             (for ([v (in-list loader)])
               (check-regexp-match #rx"WORKERS=[02],PNG_DIGEST=[0-9a-f]+,DATASET=fixture$" v))
             (for ([v (in-list resnet)])
               (check-regexp-match #rx",DATASET=fixture,BASE=4$" v)))))

        (test-case "the loader stage fails, and measures nothing, when the PNG tree cannot be written"
          (in-fresh-dir
           (lambda (dir)
             (check-not-equal? (run-baseline dir "loader" #:folder-fails? #t) 0)
             (check-equal? (bench-calls dir) '()))))))]))
