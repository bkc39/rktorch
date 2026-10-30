#lang racket/base

(module+ test
  (require (only-in racket/file make-temporary-directory make-temporary-file)
           (only-in racket/list remove-duplicates)
           (only-in racket/match match-define)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "cifar-folder.rkt" ensure-cifar10-folder!)
           (only-in "e2e-examples.rkt" find-examples)
           (only-in "e2e-suite.rkt" run-e2e)
           (only-in "fixtures/pre-hook-torch.rkt" call-with-pre-hook-torch)
           (only-in "harness.rkt" with-temporary-path)
           (only-in "pytorch.rkt" pytorch-available?)
           (only-in "summary.rkt" read-records summary-markdown))

  (define quiet (open-output-nowhere))

  (define (records-of thunk)
    (with-temporary-path (path (make-temporary-file "bench-~a.jsonl"))
      (call-with-output-file path #:exists 'truncate thunk)
      (read-records (list path))))

  (define-values (png-digest h-records refusal refusal-log pre-hook-records)
    (with-temporary-path (base (make-temporary-directory "bench-png-~a"))
      (define digest (ensure-cifar10-folder! "fixture" #:base base))
      (define env (environment-variables-copy (current-environment-variables)))
      (environment-variables-set! env #"RKTORCH_CIFAR10_PNG_DIR" (path->bytes base))
      (define (tiny pipeline workers [tree (substring digest 0 12)])
        `(,(format "~a-~a" pipeline workers)
          ("PIPELINE" . ,pipeline) ("WORKERS" . ,(number->string workers))
          ("DATASET" . "fixture") ("BASE" . "4") ("PNG_DIGEST" . ,tree)))
      (define (run names variants log)
        (parameterize ([current-environment-variables env])
          (records-of (lambda (out)
                        (run-e2e (find-examples names) #:rounds 1 #:full? #f #:variants variants
                                 #:cache #f #:max-load #f #:libtorch "x" #:out out #:log log
                                 #:time-limit 900)))))
      (define names (if (pytorch-available?)
                        '("cifar-loader" "py-cifar-loader" "cifar-resnet" "py-cifar-resnet")
                        '("cifar-loader" "cifar-resnet")))
      (define records (run names (list (tiny "per-sample" 2) (tiny "batched" 0)) quiet))
      (define log (open-output-string))
      (define refused
        (if (pytorch-available?)
            (run '("py-cifar-loader") (list (tiny "batched" 0 "0000")) log)
            '()))
      (define pre-hook
        (call-with-pre-hook-torch
         (lambda (old-torch)
           (match-define (cons name settings) (tiny "batched" 0))
           (run '("cifar-loader") (list `(,name ,old-torch ,@settings)) quiet))))
      (values digest records refused (get-output-string log) pre-hook)))

  (define (h-group g) (filter (lambda (r) (equal? (hash-ref r 'group) g)) h-records))

  (test-case "H's workloads run under e2e on the fixture PNG tree, two epochs of whole batches"
    (check-equal? (h-group "error") '())
    (define loader-runs
      (filter (lambda (r) (regexp-match? #rx"cifar-loader$" (hash-ref r 'case))) (h-group "run")))
    (check-equal? (remove-duplicates (map (lambda (r) (hash-ref r 'steps)) loader-runs)) '(4)
                  "two epochs of the 256 images in whole batches of 128")
    (check-equal? (length (h-group "summary")) (if (pytorch-available?) 8 4)))

  (test-case "the loader workload runs against a torch without this checkout's step hook"
    (check-equal? (map (lambda (r) (hash-ref r 'group)) pre-hook-records) '("run" "summary"))
    (check-equal? (hash-ref (car pre-hook-records) 'steps) 4))

  (test-case "H rows each workload, beside its twin where PyTorch is installed"
    (define md (summary-markdown h-records))
    (check-regexp-match #rx"## H\\. CIFAR-10 PNG loader" md)
    (check-regexp-match #rx"\\| loader only \\| per-sample \\| 2 \\| [a-z]+ \\| images/s \\| [0-9]+ \\|" md)
    (check-regexp-match #rx"\\| 09 ResNet \\| batched \\| 0 \\| [a-z]+ \\| s/epoch \\| [0-9.]+ \\|" md)
    (when (pytorch-available?)
      (check-regexp-match
       #rx"\\| loader only \\| per-sample \\| 2 \\| [a-z]+ \\| images/s \\| [0-9]+ \\| [0-9]+ \\| [0-9.]+ \\|"
       md)
      (check-regexp-match
       #rx"\\| 09 ResNet \\| batched \\| 0 \\| [a-z]+ \\| s/epoch \\| [0-9.]+ \\| [0-9.]+ \\| [0-9.]+ \\|"
       md)))

  (when (pytorch-available?)
    (test-case "a PyTorch twin records the tree's digest and whether it pinned its batches"
      (define twin-runs
        (filter (lambda (r) (regexp-match? #rx"^py-cifar-" (hash-ref r 'case))) (h-group "run")))
      (check-equal? (length twin-runs) 4)
      (for ([r (in-list twin-runs)])
        (check-equal? (hash-ref (hash-ref r 'meta) 'fixture_digest) png-digest)
        (check-true (boolean? (hash-ref (hash-ref r 'meta) 'pin_memory)))))

    (test-case "a PyTorch twin refuses a tree whose digest is not the one asked for"
      (check-equal? (for/list ([r (in-list refusal)]) (list (hash-ref r 'group) (hash-ref r 'case)))
                    '(("run" "py-cifar-loader") ("error" "py-cifar-loader")))
      (check-regexp-match #rx"has digest [0-9a-f]+, not 0000" refusal-log))))
