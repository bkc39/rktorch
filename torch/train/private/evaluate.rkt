#lang racket/base

(require (only-in racket/match match-define)
         (only-in (submod "../../data/loader.rkt" private) with-traversal)
         (only-in "../../foreign.rkt" with-no-grad)
         (only-in "../../nn/mode.rkt" call-with-eval-mode)
         (only-in "book.rkt" epoch-end-reached!)
         (only-in "dispatch.rkt"
                  call-hooks call-trainer note-batch-size! publish-epoch! publish-steps!
                  transfer)
         (only-in "hooks.rkt" loop-hook-names)
         (only-in "state.rkt"
                  run-book run-current-stage run-device run-trainer set-run-batch-index!
                  set-run-current-stage!))

(provide evaluate)

(define stage-loops
  #hasheq((sanity-check . validation) (validate . validation) (test . test) (predict . predict)))

(define (evaluate r loader count stage)
  (match-define (list start epoch-start batch-start step batch-end epoch-end end)
    (loop-hook-names (hash-ref stage-loops stage)))
  (define outer (run-current-stage r))
  (set-run-current-stage! r stage)
  (begin0
    (call-with-eval-mode
     (run-trainer r)
     (lambda ()
       (with-no-grad
         (call-hooks r start)
         (call-hooks r epoch-start)
         (define outputs
           (with-traversal ([_ batch] loader)
             (for/list ([i (in-range count)])
               (set-run-batch-index! r i)
               (define values-in (transfer r (run-device r) (batch i)))
               (note-batch-size! r values-in)
               (apply call-hooks r batch-start values-in)
               (define out (apply call-trainer r step values-in))
               (apply call-hooks r batch-end out values-in)
               (publish-steps! r 'eval (add1 i))
               out)))
         (set-run-batch-index! r #f)
         (epoch-end-reached! (run-book r) 'eval)
         (call-hooks r epoch-end)
         (publish-epoch! r 'eval)
         (call-hooks r end)
         outputs)))
    (set-run-current-stage! r outer)))
