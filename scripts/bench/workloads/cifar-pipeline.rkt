#lang racket/base

(require (only-in racket/contract/base ->* any/c or/c)
         (only-in racket/match match)
         (only-in racket/string string-prefix?)
         (only-in torch cpu-device flatten item select unsqueeze with-no-grad ~>)
         (only-in torch/data/loader dataloader default-collate)
         (only-in torch/vision/image-folder image-folder)
         (only-in torch/vision/transforms
                  convert-image-dtype normalize random-crop random-horizontal-flip)
         (only-in "../../../examples/test/private/env.rkt" env-number)
         (only-in "../../../torch/private/contract.rkt" define/contract-out)
         (only-in "../cifar-folder.rkt" cifar10-folder cifar10-folder-digest))

(provide cifar-settings
         log-batch!)

(define cifar-mean '(0.4914 0.4822 0.4465))
(define cifar-std '(0.2470 0.2435 0.2616))

(define (cifar-folder-root dataset expected-digest)
  (define digest (cifar10-folder-digest dataset))
  (unless digest
    (raise-user-error 'cifar-folder
                      "no PNG tree at ~a: run racket scripts/bench/cifar-folder.rkt ~a"
                      (cifar10-folder dataset) dataset))
  (unless (or (not expected-digest) (string-prefix? digest expected-digest))
    (raise-user-error 'cifar-folder "the tree at ~a has digest ~a, not ~a"
                      (cifar10-folder dataset) digest expected-digest))
  (cifar10-folder dataset))

(define (per-sample image)
  (~> image convert-image-dtype (unsqueeze 0) (random-crop #:padding 4) random-horizontal-flip
      (select 0 0) (normalize cifar-mean cifar-std)))

(define (augmented-batch items)
  (define-values (xs ys) (default-collate items))
  (values (~> xs (random-crop #:padding 4) random-horizontal-flip (normalize cifar-mean cifar-std))
          ys))

(define/contract-out (cifar-folder-loader root pipeline generator
                                         #:batch batch #:workers workers
                                         #:drop-last? [drop-last? #f])
  (->* (path-string? (or/c "per-sample" "batched") any/c
                     #:batch exact-positive-integer? #:workers exact-nonnegative-integer?)
       (#:drop-last? boolean?)
       any/c)
  (define train (build-path root "train"))
  (define-values (ds collate)
    (match pipeline
      ["per-sample" (values (image-folder train #:transform per-sample #:device (cpu-device))
                            default-collate)]
      ["batched" (values (image-folder train #:transform convert-image-dtype #:device (cpu-device))
                         augmented-batch)]))
  (dataloader ds #:batch-size batch #:shuffle? #t #:drop-last? drop-last? #:collate collate
              #:generator generator #:workers workers))

(define-logger rktorch-step)

(define (log-batch! x)
  (when (log-level? rktorch-step-logger 'debug 'rktorch-step)
    (define host-done (current-inexact-monotonic-milliseconds))
    (with-no-grad (~> x flatten (select 0 0) item))
    (log-message rktorch-step-logger 'debug 'rktorch-step "step"
                 (vector 0 (current-inexact-monotonic-milliseconds) (current-gc-milliseconds)
                         host-done)
                 #f)))

(define (cifar-settings)
  (values (env-number "EPOCHS" 2)
          (env-number "WORKERS" 0)
          (env-number "BATCH" 128)
          (or (getenv "PIPELINE") "per-sample")
          (cifar-folder-root (or (getenv "DATASET") "full") (getenv "PNG_DIGEST"))))

(module+ test
  (require (only-in racket/file make-temporary-directory)
           rackunit
           (only-in "../cifar-folder.rkt" ensure-cifar10-folder!)
           (only-in "../harness.rkt" with-temporary-path))

  (test-case "a runner takes the tree only when it is written and its digest is the one asked for"
    (with-temporary-path (base (make-temporary-directory "cifar-pipeline-~a"))
      (parameterize ([current-environment-variables
                      (environment-variables-copy (current-environment-variables))])
        (putenv "RKTORCH_CIFAR10_PNG_DIR" (path->string base))
        (check-exn #rx"no PNG tree at .*cifar-folder.rkt fixture"
                   (lambda () (cifar-folder-root "fixture" #f)))
        (define digest (ensure-cifar10-folder! "fixture" #:base base))
        (check-equal? (cifar-folder-root "fixture" (substring digest 0 12))
                      (build-path base "fixture"))
        (check-equal? (cifar-folder-root "fixture" #f) (build-path base "fixture"))
        (check-exn #rx"has digest [0-9a-f]+, not 0000"
                   (lambda () (cifar-folder-root "fixture" "0000")))))))
