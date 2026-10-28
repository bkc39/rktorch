#lang racket/base

(require (only-in (submod "../../data/loader.rkt" private) with-traversal)
         (only-in "../../foreign.rkt" with-no-grad)
         (only-in "../../nn/mode.rkt" call-with-eval-mode)
         (only-in "dispatch.rkt"
                  call-hooks call-trainer note-batch-size! publish-epoch! transfer)
         (only-in "state.rkt"
                  run-device run-stage run-trainer set-run-batch-index! set-run-stage!))

(provide evaluate)

(define (hook-name fmt part)
  (string->symbol (format fmt part)))

(define (evaluate r loader count stage part)
  (define outer (run-stage r))
  (define (named fmt) (hook-name fmt part))
  (set-run-stage! r stage)
  (begin0
    (call-with-eval-mode
     (run-trainer r)
     (lambda ()
       (with-no-grad
         (call-hooks r (named "on-~a-start"))
         (call-hooks r (named "on-~a-epoch-start"))
         (define outputs
           (with-traversal ([_ batch] loader)
             (for/list ([i (in-range count)])
               (set-run-batch-index! r i)
               (define values-in (transfer r (run-device r) (call-with-values (lambda () (batch i)) list)))
               (note-batch-size! r values-in)
               (apply call-hooks r (named "on-~a-batch-start") values-in)
               (define out (apply call-trainer r (named "~a-step") values-in))
               (apply call-hooks r (named "on-~a-batch-end") out values-in)
               out)))
         (set-run-batch-index! r #f)
         (call-hooks r (named "on-~a-epoch-end"))
         (publish-epoch! r 'eval)
         (call-hooks r (named "on-~a-end"))
         outputs)))
    (set-run-stage! r outer)))
