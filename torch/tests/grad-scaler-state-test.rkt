#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" * backward! sum tensor)
           (only-in "../nn.rkt"
                    Parameter grad-scaler grad-scaler-scale grad-scaler-scale-loss
                    grad-scaler-state-dict grad-scaler-step! grad-scaler-update!
                    load-grad-scaler-state! sgd zero-grads!))

  (define (coefficients t)
    (list (if (memv t '(2 5)) +inf.0 (+ 0.5 (* 0.1 t))) -0.25))

  (define (scaled-steps! s opt p from n)
    (for/list ([t (in-range from (+ from n))])
      (zero-grads! opt)
      (backward! (grad-scaler-scale-loss s (sum (* p (tensor (coefficients t))))))
      (grad-scaler-step! s opt)
      (grad-scaler-update! s)
      (grad-scaler-scale s)))

  (define (scaler #:init-scale [init 1024.0] #:growth-factor [growth 2.0])
    (grad-scaler #:device 'cpu #:init-scale init #:growth-factor growth #:growth-interval 2))

  (test-case "T3-L13: a round trip after a sequence with infs continues as the original"
    (define p (Parameter (tensor '(1.0 -1.0))))
    (define opt (sgd (list p) #:lr 0.1))
    (define s (scaler))
    (void (scaled-steps! s opt p 0 6))
    (define saved (grad-scaler-state-dict s))
    (check-equal? (hash-ref saved "_growth_tracker") 0 "step 5 overflowed")
    (define twin (scaler #:init-scale 8.0))
    (load-grad-scaler-state! twin saved)
    (check-equal? (grad-scaler-state-dict twin) saved)
    (define q (Parameter (tensor '(1.0 -1.0))))
    (define twin-opt (sgd (list q) #:lr 0.1))
    (check-equal? (scaled-steps! twin twin-opt q 6 4) (scaled-steps! s opt p 6 4)))

  (test-case "T3-L13: loading into a scaler that has already scaled overwrites its scale and tracker"
    (define p (Parameter (tensor '(1.0 -1.0))))
    (define opt (sgd (list p) #:lr 0.1))
    (define s (scaler))
    (void (scaled-steps! s opt p 0 3))
    (define saved (grad-scaler-state-dict s))
    (define q (Parameter (tensor '(1.0 -1.0))))
    (define twin-opt (sgd (list q) #:lr 0.1))
    (define twin (scaler #:init-scale 8.0 #:growth-factor 4.0))
    (void (scaled-steps! twin twin-opt q 0 1))
    (load-grad-scaler-state! twin saved)
    (check-equal? (grad-scaler-state-dict twin) saved)
    (check-equal? (scaled-steps! twin twin-opt q 3 4) (scaled-steps! s opt p 3 4)))

  (test-case "T3-L13: a disabled scaler's state is empty, and loading into one does nothing"
    (define off (grad-scaler #:device 'cpu #:enabled? #f))
    (check-equal? (grad-scaler-state-dict off) (hash))
    (load-grad-scaler-state! off (grad-scaler-state-dict (scaler)))
    (check-equal? (grad-scaler-state-dict off) (hash)))

  (test-case "T3-L13: loading an empty state into an enabled scaler raises"
    (check-exn exn:fail? (lambda () (load-grad-scaler-state! (scaler) (hash)))))

  (test-case "T3-L13: loading overwrites the factors and the interval"
    (define s (scaler))
    (load-grad-scaler-state! s (hash "scale" 4.0 "growth_factor" 3.0 "backoff_factor" 0.25
                                     "growth_interval" 1 "_growth_tracker" 0))
    (define p (Parameter (tensor '(1.0 -1.0))))
    (define opt (sgd (list p) #:lr 0.1))
    (check-equal? (scaled-steps! s opt p 0 3) '(12.0 36.0 9.0))))
