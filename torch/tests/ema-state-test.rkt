#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" backward! copy! make-generator randn with-no-grad)
           (only-in "../nn.rkt"
                    Linear ema ema-average ema-n-averaged ema-update! mse-loss
                    parameters set-ema-n-averaged! sgd step! zero-grads!)
           (only-in "private/comparable.rkt" layer-bytes))

  (define g (make-generator 9))
  (define x (randn 4 3 #:generator g))
  (define y (randn 4 1 #:generator g))

  (define (train! opt net)
    (zero-grads! opt)
    (backward! (mse-loss (net x) y))
    (step! opt))

  (test-case "T3-L14: n updates give n"
    (define net (Linear 3 1))
    (define e (ema net (Linear 3 1) #:decay 0.5))
    (check-equal? (ema-n-averaged e) 0)
    (for ([_ (in-range 3)]) (ema-update! e))
    (check-equal? (ema-n-averaged e) 3)
    (set-ema-n-averaged! e 7)
    (check-equal? (ema-n-averaged e) 7))

  (test-case "T3-L14: a rebuilt EMA with the restored count and weights matches the original"
    (define net (Linear 3 1))
    (define opt (sgd (parameters net) #:lr 0.1))
    (define e (ema net (Linear 3 1) #:decay 0.5))
    (for ([_ (in-range 3)]) (train! opt net) (ema-update! e))
    (define rebuilt (ema net (Linear 3 1) #:decay 0.5))
    (with-no-grad
      (for ([p (in-list (parameters (ema-average rebuilt)))]
            [q (in-list (parameters (ema-average e)))])
        (copy! p q)))
    (set-ema-n-averaged! rebuilt (ema-n-averaged e))
    (for ([_ (in-range 2)])
      (train! opt net)
      (ema-update! e)
      (ema-update! rebuilt))
    (check-equal? (layer-bytes (ema-average rebuilt)) (layer-bytes (ema-average e)))
    (check-equal? (ema-n-averaged rebuilt) 5)))
