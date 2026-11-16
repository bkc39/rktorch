#lang racket/base

(require (only-in racket/contract/base -> hash/c list/c listof)
         (only-in racket/date current-date date->string date-display-format)
         (only-in racket/list fourth last)
         (only-in racket/match match match-define match-lambda)
         (only-in racket/port with-output-to-string)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-trim)
         (only-in racket/system system*)
         (only-in torch device-type device? torch-version)
         (only-in torch/scribblings/examples/20-fashion-mnist fashion-recipe)
         (only-in "../examples/test/private/fashion-mnist-kit.rkt" device-name precision-for)
         (only-in "../torch/private/contract.rkt" define/contract-out))

(define-runtime-path root "..")

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
  (match (device-type device)
    ['cuda (or (read-of "nvidia-smi" "--query-gpu=name" "--format=csv,noheader") "cuda")]
    [_ (device-name device)]))

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

(define/contract-out (fashion-mnist-record racket-rows lightning-rows device
                                           #:config config #:versions versions
                                           #:commit commit #:seconds seconds)
  (-> (listof row/c) (listof row/c) device?
      #:config hash? #:versions (hash/c symbol? string?)
      #:commit string? #:seconds (list/c real? real?)
      hash?)
  (match-define (list racket-seconds lightning-seconds) seconds)
  (hasheq 'model "two conv blocks (3x3 conv, batch norm, ReLU, 2x2 max pool), Linear 1568-128-10"
          'data "Fashion-MNIST: 60000 train, 10000 test"
          'config config
          'device (device-name device)
          'hardware (hardware device)
          'precision (symbol->string (precision-for device))
          'torch (torch-version)
          'racket-version (version)
          'python (hash-ref versions 'python)
          'python-torch (hash-ref versions 'torch)
          'cuda (hash-ref versions 'cuda)
          'lightning-version (hash-ref versions 'lightning)
          'seed (hash-ref fashion-recipe 'seed)
          'commit commit
          'command (command device (length racket-rows))
          'date (parameterize ([date-display-format 'iso-8601])
                  (date->string (current-date)))
          'racket (side racket-rows racket-seconds)
          'lightning (side lightning-rows lightning-seconds)))

(module+ main
  (require (only-in json jsexpr->string)
           (only-in racket/pretty pretty-write)
           (only-in torch/scribblings/examples/20-fashion-mnist warmup-steps)
           (only-in "../examples/test/private/env.rkt" env-number)
           (only-in "../examples/test/private/fashion-mnist-kit.rkt"
                    full-batch-size headline-device print-epoch-rows run-headline twin-config
                    twin-json))
  (define-runtime-path out "../torch/scribblings/results/fashion-mnist.rktd")
  (define device (headline-device))
  (define epochs (env-number "EPOCHS" 5))
  (define commit (read-of "git" "describe" "--always" "--dirty" "--abbrev=12"))
  (define (timed thunk)
    (define start (current-inexact-milliseconds))
    (define result (call-with-values thunk list))
    (values result (/ (- (current-inexact-milliseconds) start) 1000.0)))
  (define-values (headline racket-seconds)
    (timed (lambda () (run-headline #:epochs epochs #:device device))))
  (match-define (list racket-rows _run total) headline)
  (define-values (twins lightning-seconds)
    (timed (lambda () (twin-json device (precision-for device) 'full epochs))))
  (match-define (list twin) twins)
  (define twin-config-seen (hash-ref twin 'config))
  (define expected
    (twin-config #:batch full-batch-size #:epochs epochs #:total total
                 #:warmup (warmup-steps total)))
  (for ([(key value) (in-hash expected)])
    (unless (equal? (hash-ref twin-config-seen key) value)
      (raise-user-error 'fashion-mnist-results "the twin's ~a is ~s, not ~s\n~a"
                        key (hash-ref twin-config-seen key) value
                        (jsexpr->string twin-config-seen))))
  (define record
    (fashion-mnist-record racket-rows (hash-ref twin 'epochs) device
                          #:config expected #:versions (hash-ref twin 'versions)
                          #:commit commit #:seconds (list racket-seconds lightning-seconds)))
  (call-with-output-file out #:exists 'truncate
    (lambda (port) (pretty-write record port)))
  (print-epoch-rows racket-rows)
  (printf "Lightning: test acc ~a\n" (hash-ref (hash-ref record 'lightning) 'accuracy))
  (printf "wrote ~a\n" out))
