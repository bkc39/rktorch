#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../main.rkt"
                    * backward! grad has-grad? ones sum tensor tensor->list)
           (only-in "../nn.rkt" Linear Parameter Sequential adam parameters sgd
                    step! step-lr zero-grads!))

  (define (crossings-of thunk)
    (define here (current-thread))
    (define n 0)
    (with-crossing-hook (lambda () (when (eq? (current-thread) here) (set! n (add1 n))))
      (thunk))
    n)

  (define (backed-up width depth)
    (define net (apply Sequential (for/list ([_ (in-range depth)]) (Linear width width))))
    (backward! (sum (net (ones 2 width))))
    net)

  (test-case "zero-grads! crosses into native code once, whatever the parameter count"
    (for ([depth (in-list '(1 4 16))])
      (define net (backed-up 3 depth))
      (define opt (sgd (parameters net) #:lr 0.1))
      (check-equal? (length (parameters net)) (* 2 depth))
      (check-equal? (crossings-of (lambda () (zero-grads! opt))) 1
                    (format "~a parameters" (* 2 depth)))))

  (test-case "after zero-grads! no parameter has a gradient"
    (define net (backed-up 3 2))
    (define opt (adam (parameters net) #:lr 0.1))
    (check-true (andmap has-grad? (parameters net)))
    (zero-grads! opt)
    (check-false (ormap has-grad? (parameters net)))
    (zero-grads! opt)
    (check-false (ormap has-grad? (parameters net)) "a second reset is harmless"))

  (test-case "#:set-to-none? #f zeroes every gradient in place"
    (define net (backed-up 3 2))
    (define opt (sgd (parameters net) #:lr 0.1))
    (zero-grads! opt #:set-to-none? #f)
    (for ([p (in-list (parameters net))])
      (check-true (has-grad? p))
      (check-true (andmap zero? (tensor->list (grad p))))))

  (test-case "a backward after the reset gives a fresh backward's gradient"
    (define w (Parameter (tensor '(1.0 -2.0 3.0))))
    (define opt (sgd (list w) #:lr 0.1))
    (define (back!) (backward! (sum (* w w))))
    (back!)
    (back!)
    (define held (grad w))
    (check-equal? (tensor->list held) '(4.0 -8.0 12.0) "gradients accumulate")
    (zero-grads! opt)
    (back!)
    (check-equal? (tensor->list (grad w)) '(2.0 -4.0 6.0))
    (check-equal? (tensor->list held) '(4.0 -8.0 12.0)
                  "a gradient taken before the reset is not written by the next backward"))

  (test-case "a step after the reset leaves a parameter without a gradient alone"
    (define w (Parameter (tensor '(1.0 2.0))))
    (define opt (sgd (list w) #:lr 0.1))
    (backward! (sum w))
    (zero-grads! opt)
    (step! opt)
    (check-equal? (tensor->list w) '(1.0 2.0)))

  (test-case "Adam's bias correction counts the steps a parameter had a gradient for"
    (define a (Parameter (tensor '(1.0))))
    (define b (Parameter (tensor '(1.0))))
    (define opt (adam (list a b) #:lr 0.1))
    (backward! (sum a))
    (step! opt)
    (zero-grads! opt)
    (backward! (sum (* a b)))
    (step! opt)
    (check-= (car (tensor->list b)) 0.9 1e-6 "b's first step moves it by lr"))

  (test-case "a schedule drops its optimizer's gradients"
    (define w (Parameter (tensor '(1.0 2.0))))
    (define s (step-lr (sgd (list w) #:lr 0.1) #:step-size 1))
    (backward! (sum w))
    (zero-grads! s)
    (check-false (has-grad? w))))
