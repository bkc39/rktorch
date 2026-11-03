#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt" collecting-finalized collecting-logger))

  (define (new-tally) (make-hasheq))

  (define (note! t key v)
    (hash-update! t key (lambda (vs) (append vs (list v))) '()))

  (define (notes t key) (hash-ref t key '()))

  (define (ones-loader n)
    (dataloader (tensor-dataset (ones n 1) (ones n 1)) #:batch-size 1))

  (define-trainer Plain (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-test-step (x y) (mse-loss (net x) y)))

  (define planned (exn:fail "planned" (current-continuation-marks)))

  (define-trainer Failing (net t)
    (define-init (tl) (set! net (Linear 1 1)) (set! t tl))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (raise planned))
    (define-hook (teardown stage) (note! t 'teardown stage)))

  (define-callback breaking (t)
    (define-hook (on-exception e) (error 'breaking "inside on-exception")))

  (define-callback catching (t)
    (define-hook (on-exception e) (note! t 'caught e))
    (define-hook (teardown stage) (note! t 'teardown stage)))

  (test-case "exceptions: re-raised eq?, every on-exception and teardown run once"
    (define t (new-tally))
    (define l (collecting-logger))
    (define receiver (make-log-receiver (current-logger) 'debug 'rktorch-train))
    (define raised
      (with-handlers ([values values])
        (fit (Failing t) (ones-loader 2) #:max-epochs 1 #:logger l
             #:callbacks (list (breaking t) (catching t)))))
    (check-eq? raised planned)
    (check-equal? (notes t 'caught) (list planned))
    (check-equal? (notes t 'teardown) '(fit fit))
    (check-equal? (collecting-finalized l) '(failed))
    (check-not-false (sync/timeout 0 receiver)))

  (define-callback interrupting (t)
    (define-hook (on-exception e)
      (note! t 'caught e)
      (break-thread (current-thread))))

  (test-case "exceptions: a break inside on-exception is absorbed after an exn:fail"
    (define t (new-tally))
    (define raised
      (with-handlers ([exn:break? values])
        (begin0 (with-handlers ([exn:fail? values])
                  (fit (Failing t) (ones-loader 2) #:max-epochs 1 #:logger #f
                       #:callbacks (list (interrupting t))))
                (sleep 0))))
    (check-eq? raised planned)
    (check-equal? (notes t 'caught) (list planned))
    (check-equal? (notes t 'teardown) '(fit)))

  (test-case "exceptions: with breaks disabled, a break inside on-exception stays pending"
    (define t (new-tally))
    (define-values (raised broke?)
      (parameterize-break #f
        (let ([raised (with-handlers ([exn:fail? values])
                        (fit (Failing t) (ones-loader 2) #:max-epochs 1 #:logger #f
                             #:callbacks (list (interrupting t))))])
          (values raised
                  (with-handlers ([exn:break? (lambda (e) #t)])
                    (parameterize-break #t (sleep 0))
                    #f)))))
    (check-eq? raised planned)
    (check-true broke?))

  (struct finalize-failing ([statuses #:mutable])
    #:methods gen:training-logger
    [(define (logger-log-hyperparams! l params) (void))
     (define (logger-log-metrics! l metrics step) (void))
     (define (logger-finalize! l status)
       (set-finalize-failing-statuses! l (cons status (finalize-failing-statuses l)))
       (when (eq? status 'success) (error 'finalize "planned finalize failure")))])

  (test-case "exceptions: a failing success finalize is not finalized again as failed"
    (define l (finalize-failing '()))
    (check-exn #rx"planned finalize failure"
               (lambda () (fit (Plain) (ones-loader 2) #:max-epochs 1 #:logger l)))
    (check-equal? (finalize-failing-statuses l) '(success))
    (define m (finalize-failing '()))
    (check-exn #rx"planned finalize failure"
               (lambda () (test (Plain) (ones-loader 2) #:logger m)))
    (check-equal? (finalize-failing-statuses m) '(success))))
