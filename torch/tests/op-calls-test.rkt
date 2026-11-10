#lang racket/base

(module+ test
  (require rackunit
           (only-in "private/wedge.rkt" exit-status-of))

  (define (with-one-reaper thunk)
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"RKTORCH_REAPERS" #"1")
    (parameterize ([current-environment-variables env])
      (thunk)))

  (define dead-generator-program
    '(begin
       (require (only-in torch make-generator zeros)
                (only-in torch/foreign/raw/cas cas-add! cas-claim!)
                (only-in torch/foreign/raw/race-points race-hook)
                (only-in torch/runtime/reaper
                         reap-after-collection! reap-when-dead! reaped wake-reapers!)
                (only-in (submod torch/runtime/reaper private)
                         reaper-statistics reaper-threads)
                (only-in torch/tests/private/op-calls op-calls-of)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define holding (box #f))
       (define go (box #f))
       (set-box! race-hook
                 (lambda (name subject)
                   (when (and (eq? name 'reaper-taken)
                              (memq subject (reaper-threads))
                              (cas-claim! holding #f subject #:site 'gate))
                     (let spin () (unless (unbox go) (spin))))))
       (define kind (reaped 'held (lambda (_v) (void))))
       (define (released) (hash-ref (reaper-statistics) 'released))
       (define (settle!)
         (collect-garbage)
         (let reap ([seen (released)])
           (reap-after-collection!)
           (unless (= seen (released)) (reap (released)))))
       (define (op)
         (reap-after-collection!)
         (zeros 1))
       (call-with-process-deadline
        50
        (lambda ()
          (let hold ()
            (unless (unbox holding)
              (reap-when-dead! kind (make-vector 1 0))
              (collect-garbage)
              (wake-reapers!)
              (sleep 0.01)
              (hold)))
          (op-calls-of op)
          (settle!)
          (define alone (op-calls-of op))
          (settle!)
          (make-generator 1)
          (collect-garbage)
          (define before (released))
          (define with-a-dead-generator (op-calls-of op))
          (define freed (- (released) before))
          (set-box! go #t)
          (eprintf "op alone ~a, with a dead generator freed in its count ~a (~a released)\n"
                   alone with-a-dead-generator freed)
          (exit (if (and (= with-a-dead-generator alone) (= freed 1)) 0 1))))))

  (test-case "op-calls-of counts no release a reaper batch on its thread makes"
    (check-equal? (with-one-reaper (lambda () (exit-status-of dead-generator-program #:within 60)))
                  0)))
