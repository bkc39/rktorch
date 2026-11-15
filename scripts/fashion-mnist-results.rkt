#lang racket/base

(require (only-in racket/contract/base -> list/c listof)
         (only-in racket/date current-date date->string date-display-format)
         (only-in racket/list fourth last)
         (only-in racket/match match match-define match-lambda)
         (only-in racket/port with-output-to-string)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-trim)
         (only-in racket/system system*)
         (only-in torch device-type device? torch-version)
         (only-in torch/scribblings/examples/20-fashion-mnist
                  accumulate batch-size clip-val lr precision-for seed weight-decay)
         (only-in "../torch/private/contract.rkt" define/contract-out))

(define-runtime-path root "..")

(define lightning-version "2.6.5")

(define row/c (list/c exact-positive-integer? real? real? real? real?))

(define (read-of program . args)
  (match (find-executable-path program)
    [#f #f]
    [exe (string-trim
          (with-output-to-string
            (lambda ()
              (parameterize ([current-directory root])
                (apply system* exe args)))))]))

(define (rounded x digits)
  (define scale (expt 10.0 digits))
  (/ (round (* scale x)) scale))

(define (hardware device)
  (define type (symbol->string (device-type device)))
  (match type
    ["cuda" (or (read-of "nvidia-smi" "--query-gpu=name" "--format=csv,noheader") type)]
    [_ type]))

(define (command device epochs)
  (format "EPOCHS=~a nix develop ~a--command racket scripts/fashion-mnist-results.rkt"
          epochs
          (if (eq? (device-type device) 'cuda) ".#cuda " "")))

(define (side rows seconds)
  (define kept
    (map (match-lambda
           [(list epoch train-loss val-loss acc s)
            (list epoch (rounded train-loss 4) (rounded val-loss 4) (rounded acc 4)
                  (rounded s 1))])
         rows))
  (hasheq 'seconds (rounded seconds 1) 'accuracy (fourth (last kept)) 'epochs kept))

(define (fashion-mnist-config epochs)
  (hasheq 'lr lr 'weight-decay weight-decay 'clip clip-val 'accumulate accumulate
          'batch (hash-ref batch-size 'full) 'epochs epochs))

(define/contract-out (fashion-mnist-record racket-rows lightning-rows device
                                           #:commit commit #:seconds seconds)
  (-> (listof row/c) (listof row/c) device? #:commit string? #:seconds (list/c real? real?) hash?)
  (match-define (list racket-seconds lightning-seconds) seconds)
  (hasheq 'model "two conv blocks (3x3 conv, batch norm, ReLU, 2x2 max pool), Linear 1568-128-10"
          'data "Fashion-MNIST: 60000 train, 10000 test"
          'config (fashion-mnist-config (length racket-rows))
          'device (symbol->string (device-type device))
          'hardware (hardware device)
          'precision (symbol->string (precision-for device))
          'torch (torch-version)
          'lightning-version lightning-version
          'seed seed
          'commit commit
          'command (command device (length racket-rows))
          'date (parameterize ([date-display-format 'iso-8601])
                  (date->string (current-date)))
          'racket (side racket-rows racket-seconds)
          'lightning (side lightning-rows lightning-seconds)))

(module+ main
  (require (only-in json jsexpr->string)
           (only-in racket/pretty pretty-write)
           (only-in torch/private/util cache-dir)
           (only-in torch/scribblings/examples/20-fashion-mnist pick-device train-fashion-mnist)
           (only-in torch/tests/private/python-env call-with-python-env python-result)
           (only-in "../examples/test/private/env.rkt" env-number))
  (define-runtime-path out "../torch/scribblings/results/fashion-mnist.rktd")
  (define device (pick-device))
  (define epochs (env-number "EPOCHS" 5))
  (define commit (read-of "git" "describe" "--always" "--dirty" "--abbrev=12"))
  (define (timed thunk)
    (define start (current-inexact-milliseconds))
    (define result (thunk))
    (values result (/ (- (current-inexact-milliseconds) start) 1000.0)))
  (define-values (racket-rows racket-seconds)
    (timed (lambda ()
             (define-values (rows _run) (train-fashion-mnist #:epochs epochs #:device device))
             rows)))
  (define-values (twin lightning-seconds)
    (timed (lambda ()
             (call-with-python-env
              #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string (device-type device)))
                          (cons "RKTORCH_PARITY_PRECISION"
                                (symbol->string (precision-for device)))
                          (cons "RKTORCH_TWIN_MODE" "full")
                          (cons "EPOCHS" (number->string epochs))
                          (cons "RKTORCH_FASHION_MNIST_DIR"
                                (path->string (cache-dir "RKTORCH_FASHION_MNIST_DIR"
                                                         "fashion-mnist"))))
              (lambda () (python-result "python/20_fashion_mnist.py"))))))
  (define twin-config (hash-ref twin 'config))
  (define expected
    (hasheq 'lr lr 'wd weight-decay 'clip clip-val 'A accumulate
            'batch (hash-ref batch-size 'full) 'epochs epochs 'seed seed
            'lightning lightning-version))
  (for ([(key value) (in-hash expected)])
    (unless (equal? (hash-ref twin-config key) value)
      (raise-user-error 'fashion-mnist-results "the twin's ~a is ~s, not ~s\n~a"
                        key (hash-ref twin-config key) value (jsexpr->string twin-config))))
  (define record
    (fashion-mnist-record racket-rows (hash-ref twin 'epochs) device
                          #:commit commit #:seconds (list racket-seconds lightning-seconds)))
  (call-with-output-file out #:exists 'truncate
    (lambda (port) (pretty-write record port)))
  (for ([name (in-list '(racket lightning))])
    (printf "~a: acc ~a, ~a s\n" name (hash-ref (hash-ref record name) 'accuracy)
            (hash-ref (hash-ref record name) 'seconds)))
  (printf "wrote ~a\n" out))
