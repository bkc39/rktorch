#lang racket/base

(module+ test
  (require rackunit
           syntax/parse/define
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../foreign/raw/fault.rkt" native-faulted reset-native-fault-latch!)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear parameters sgd)
           "../train.rkt")

  (define-syntax-parse-rule (with-latch-cleared-after body:expr ...+)
    (dynamic-wind void (lambda () body ...) reset-native-fault-latch!))

  (define planned (exn:fail "planned" (current-continuation-marks)))

  (define (segfault)
    (raise (exn:fail "invalid memory reference.  Some debugging context lost"
                     (current-continuation-marks))))

  (define-trainer Faulting (net place)
    (define-init (p) (set! net (Linear 1 1)) (set! place p))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (raise planned))
    (define-hook (teardown stage) (when (eq? place 'teardown) (segfault))))

  (struct faulting-logger ()
    #:methods gen:training-logger
    [(define (logger-log-hyperparams! l params) (void))
     (define (logger-log-metrics! l metrics step) (void))
     (define (logger-finalize! l status) (segfault))])

  (define (raising-callback thunk)
    (lambda-callback #:on-exception (lambda (r e) (thunk))))

  (define (fit-faulting place)
    (with-handlers ([values values])
      (fit (Faulting place)
           (dataloader (tensor-dataset (ones 2 1) (ones 2 1)) #:batch-size 1)
           #:max-epochs 1
           #:logger (and (eq? place 'logger-finalize!) (faulting-logger))
           #:callbacks (list (raising-callback
                              (case place
                                [(on-exception) segfault]
                                [(error) (lambda () (error 'hook "boom"))]
                                [else void]))))))

  (test-case "a native fault during failure recovery latches before it is suppressed"
    (for ([place (in-list '(on-exception logger-finalize! teardown))])
      (with-latch-cleared-after
        (check-false (native-faulted))
        (check-eq? (fit-faulting place) planned "the training exception still reaches the caller")
        (check-equal? (native-faulted)
                      (format "running ~a during failure recovery" place))
        (check-exn #rx"^rktorch: native code faulted earlier" (lambda () (ones 2 2))
                   "the library is disabled after the fault"))))

  (test-case "an ordinary failure during recovery latches nothing"
    (with-latch-cleared-after
      (check-eq? (fit-faulting 'error) planned)
      (check-false (native-faulted)))))
