#lang racket/base

;; Regenerates torch/scribblings/results/finetune.rktd, the table the
;; guide's fine-tuning chapter shows: examples/racket/15-finetune.rkt on the
;; full ants-and-bees set, every epoch of both phases.
;;
;;   nix develop .#cuda --command racket scripts/finetune-results.rkt

(require (only-in racket/date current-date date->string date-display-format)
         (only-in racket/format ~r)
         (only-in racket/pretty pretty-write)
         racket/runtime-path
         torch
         "../examples/racket/15-finetune.rkt")

(define-runtime-path out "../torch/scribblings/results/finetune.rktd")

(define device (pick-device))
(define start (current-inexact-milliseconds))
(define-values (records _net) (finetune #:device device))
(define total (/ (- (current-inexact-milliseconds) start) 1000.0))

(for ([r (in-list records)])
  (printf "~a epoch ~a: loss ~a, val acc ~a, ~as\n"
          (list-ref r 0) (list-ref r 1)
          (~r (list-ref r 2) #:precision '(= 4))
          (~r (list-ref r 3) #:precision '(= 4))
          (~r (list-ref r 4) #:precision '(= 1))))
(printf "total ~as\n" (~r total #:precision '(= 1)))

(call-with-output-file out #:exists 'truncate
  (lambda (port)
    (pretty-write
     (hasheq 'model "resnet18, torchvision IMAGENET1K_V1, a fresh 2-way head"
             'data "hymenoptera_data: 244 train, 153 val"
             'device (format "~a" (device-type device))
             'torch (torch-version)
             'seed 0
             'date (parameterize ([date-display-format 'iso-8601])
                     (date->string (current-date)))
             'seconds (/ (round (* 10 total)) 10)
             'epochs
             (for/list ([r (in-list records)])
               (list (list-ref r 0) (list-ref r 1)
                     (/ (round (* 1e4 (list-ref r 2))) 1e4)
                     (/ (round (* 1e4 (list-ref r 3))) 1e4)
                     (/ (round (* 10 (list-ref r 4))) 10))))
     port)))
(printf "wrote ~a\n" out)
