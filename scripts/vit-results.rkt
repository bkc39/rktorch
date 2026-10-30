#lang racket/base

;; Regenerates torch/scribblings/results/vit.rktd, the numbers the Vision
;; Transformer chapter shows: torch/scribblings/examples/18-vit.rkt on full
;; CIFAR-10, every epoch. EPOCHS overrides the chapter's count.
;;
;;   nix develop .#cuda --command racket scripts/vit-results.rkt

(require (only-in racket/date current-date date->string date-display-format)
         (only-in racket/list last third)
         (only-in racket/match match-lambda)
         (only-in racket/port with-output-to-string)
         (only-in racket/pretty pretty-write)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-trim)
         (only-in racket/system system*)
         (only-in torch device-type torch-version)
         (only-in torch/scribblings/examples/18-vit pick-device train-cifar10))

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

(define epochs (string->number (or (getenv "EPOCHS") "300")))
(define device (pick-device))
(define commit (read-of "git" "describe" "--always" "--dirty" "--abbrev=12"))
(define gpu
  (and (eq? (device-type device) 'cuda)
       (read-of "nvidia-smi" "--query-gpu=name" "--format=csv,noheader")))
(define start (current-inexact-milliseconds))
(define rows (train-cifar10 #:epochs epochs #:device device))
(define total (/ (- (current-inexact-milliseconds) start) 1000.0))

(call-with-output-file out #:exists 'truncate
  (lambda (port)
    (pretty-write
     (hasheq 'model "ViT, patch 4, width 256, depth 6, 8 heads, dropout 0.1"
             'data "CIFAR-10: 50000 train, 10000 test"
             'device (format "~a" (device-type device))
             'gpu gpu
             'torch (torch-version)
             'seed 0
             'commit commit
             'command (format "EPOCHS=~a racket scripts/vit-results.rkt" epochs)
             'date (parameterize ([date-display-format 'iso-8601])
                     (date->string (current-date)))
             'seconds (rounded total 1)
             'accuracy (rounded (third (last rows)) 4)
             'best (rounded (apply max (map third rows)) 4)
             'epochs
             (map (match-lambda
                    [(list epoch loss acc seconds)
                     (list epoch (rounded loss 4) (rounded acc 4)
                           (rounded seconds 1))])
                  rows))
     port)))
(printf "test acc ~a\n" (third (last rows)))
(printf "total ~a s\n" (rounded total 1))
(printf "wrote ~a\n" out)
