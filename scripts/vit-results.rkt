#lang racket/base

;; Regenerates torch/scribblings/results/vit.rktd, the numbers the Vision
;; Transformer chapter shows: torch/scribblings/examples/18-vit.rkt on full
;; CIFAR-10, every epoch. EPOCHS overrides the chapter's count.
;;
;;   nix develop .#cuda --command racket scripts/vit-results.rkt

(require (only-in racket/date current-date date->string date-display-format)
         (only-in racket/list last third)
         (only-in racket/match match match-lambda)
         (only-in racket/port with-output-to-string)
         (only-in racket/pretty pretty-write)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-trim)
         (only-in racket/system system*)
         (only-in torch device-type torch-version)
         (only-in torch/scribblings/examples/18-vit
                  pick-device precision seed train-cifar10 vit-config)
         (only-in "../examples/test/private/env.rkt" env-number))

(provide vit-record)

(define-runtime-path out "../torch/scribblings/results/vit.rktd")
(define-runtime-path root "..")

(define (read-of program . args)
  (string-trim
   (with-output-to-string
     (lambda ()
       (parameterize ([current-directory root])
         (apply system* (find-executable-path program) args))))))

(define (rounded x digits)
  (define scale (expt 10.0 digits))
  (/ (round (* scale x)) scale))

(define (hardware device)
  (match (device-type device)
    ['cuda (read-of "nvidia-smi" "--query-gpu=name" "--format=csv,noheader")]
    [type (symbol->string type)]))

(define (command device epochs)
  (format "EPOCHS=~a nix develop ~a--command racket scripts/vit-results.rkt"
          epochs
          (if (eq? (device-type device) 'cuda) ".#cuda " "")))

(define (vit-record rows device seconds #:commit commit)
  (hasheq 'model vit-config
          'data "CIFAR-10: 50000 train, 10000 test"
          'device (symbol->string (device-type device))
          'hardware (hardware device)
          'precision (symbol->string precision)
          'torch (torch-version)
          'seed seed
          'commit commit
          'command (command device (length rows))
          'date (parameterize ([date-display-format 'iso-8601])
                  (date->string (current-date)))
          'seconds (rounded seconds 1)
          'accuracy (rounded (third (last rows)) 4)
          'best (rounded (apply max (map third rows)) 4)
          'epochs
          (map (match-lambda
                 [(list epoch loss acc seconds)
                  (list epoch (rounded loss 4) (rounded acc 4)
                        (rounded seconds 1))])
               rows)))

(module+ main
  (define device (pick-device))
  (define commit (read-of "git" "describe" "--always" "--dirty" "--abbrev=12"))
  (define start (current-inexact-milliseconds))
  (define-values (rows _classifier)
    (train-cifar10 #:epochs (env-number "EPOCHS" 300) #:device device))
  (define total (/ (- (current-inexact-milliseconds) start) 1000.0))
  (call-with-output-file out #:exists 'truncate
    (lambda (port)
      (pretty-write (vit-record rows device total #:commit commit) port)))
  (printf "test acc ~a\n" (third (last rows)))
  (printf "total ~a s\n" (rounded total 1))
  (printf "wrote ~a\n" out))
