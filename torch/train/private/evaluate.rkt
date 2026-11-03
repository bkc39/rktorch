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
                  run-book run-current-stage run-device run-eval-batches run-global-step
                  run-stepped? run-trainer set-run-batch-index! set-run-current-stage!
                  set-run-eval-batches!))

(provide evaluate)

(define stage-loops
  #hasheq((sanity-check . validation) (validate . validation) (test . test) (predict . predict)))

(define (step-label r)
  (if (run-stepped? r) (sub1 (run-global-step r)) (run-global-step r)))

(define (eval-batch! r stage step batch-start batch-end values-in)
  (note-batch-size! r values-in)
  (apply call-hooks r batch-start values-in)
  (define out (apply call-trainer r step values-in))
  (apply call-hooks r batch-end out values-in)
  (unless (eq? stage 'sanity-check)
    (publish-steps! r 'eval (run-eval-batches r))
    (set-run-eval-batches! r (add1 (run-eval-batches r))))
  out)

(define (eval-batches! r loader count stage step batch-start batch-end)
  (define (run-batch! i batch)
    (set-run-batch-index! r i)
    (eval-batch! r stage step batch-start batch-end (transfer r (run-device r) (batch i))))
  (with-traversal ([_ batch] loader)
    (if (eq? stage 'predict)
        (for/list ([i (in-range count)])
          (run-batch! i batch))
        (for ([i (in-range count)])
          (run-batch! i batch)))))

(define (evaluate r loader count stage)
  (match-define (list start epoch-start batch-start step batch-end epoch-end end)
    (loop-hook-names (hash-ref stage-loops stage)))
  (define outer (run-current-stage r))
  (dynamic-wind
   (lambda () (set-run-current-stage! r stage))
   (lambda ()
     (call-with-eval-mode
      (run-trainer r)
      (lambda ()
        (with-no-grad
          (call-hooks r start)
          (call-hooks r epoch-start)
          (define outputs (eval-batches! r loader count stage step batch-start batch-end))
          (set-run-batch-index! r #f)
          (epoch-end-reached! (run-book r) 'eval)
          (call-hooks r epoch-end)
          (publish-epoch! r 'eval (step-label r))
          (call-hooks r end)
          outputs))))
   (lambda () (set-run-current-stage! r outer))))
