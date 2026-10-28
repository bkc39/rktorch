#lang racket/base

(module+ test
  (require rackunit
           (only-in (submod "../data/loader.rkt" private) with-traversal)
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt")

  (define (ones-loader n)
    (dataloader (tensor-dataset (ones n 1) (ones n 1)) #:batch-size 1))

  (define planned (exn:fail "planned" (current-continuation-marks)))

  (define-trainer Staged (net fail-validation?)
    (define-init ([f #f]) (set! net (Linear 1 1)) (set! fail-validation? f))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y)
      (when fail-validation? (raise planned))
      (mse-loss (net x) y))
    (define-test-step (x y) (mse-loss (net x) y))
    (define-predict-step (x y) (net x)))

  (define (stage-recorder seen)
    (define ((note name) r stage)
      (set-box! seen (cons (list name stage (run-stage r)) (unbox seen))))
    (lambda-callback #:setup (note 'setup) #:teardown (note 'teardown)))

  (test-case "run-stage names the entry during setup and teardown of test and predict"
    (define seen (box '()))
    (test (Staged) (ones-loader 2) #:logger #f #:callbacks (list (stage-recorder seen)))
    (predict (Staged) (ones-loader 2) #:callbacks (list (stage-recorder seen)))
    (check-equal? (reverse (unbox seen))
                  '((setup test test) (teardown test test)
                    (setup predict predict) (teardown predict predict))))

  (test-case "a failing validation loop leaves run-stage at train for teardown"
    (define seen (box '()))
    (define raised
      (with-handlers ([exn:fail? values])
        (fit (Staged #t) (ones-loader 2) #:val (ones-loader 1) #:max-epochs 1
             #:num-sanity-val-steps 0 #:logger #f
             #:callbacks (list (stage-recorder seen)))))
    (check-eq? raised planned)
    (check-equal? (reverse (unbox seen)) '((setup fit train) (teardown fit train))))

  (define-trainer Interrupted (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (parameterize-break #f
        (break-thread (current-thread))
        (raise planned))))

  (test-case "a break pending when the loop raises waits for on-exception and is absorbed"
    (define caught (box '()))
    (define recording
      (lambda-callback #:on-exception (lambda (r e) (set-box! caught (cons e (unbox caught))))))
    (define raised
      (with-handlers ([(lambda (e) #t) values])
        (fit (Interrupted) (ones-loader 2) #:max-epochs 1 #:logger #f
             #:callbacks (list recording))))
    (check-eq? raised planned)
    (check-equal? (unbox caught) (list planned)))

  (test-case "with-traversal runs its body with the caller's break state"
    (for* ([workers (in-list '(0 2))]
           [enabled? (in-list '(#t #f))])
      (define loader
        (dataloader (tensor-dataset (ones 4 1)) #:batch-size 1 #:workers workers))
      (define seen
        (parameterize-break enabled?
          (with-traversal ([count batch] loader)
            (batch 0)
            (list count (break-enabled)))))
      (check-equal? seen (list 4 enabled?) (format "workers=~a" workers)))))
