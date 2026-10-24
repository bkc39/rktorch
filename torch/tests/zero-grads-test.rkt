#lang racket/base

(module+ test
  (require (for-syntax racket/base syntax/parse)
           rackunit
           (only-in racket/contract exn:fail:contract:blame?)
           (only-in syntax/parse/define define-syntax-parse-rule)
           (only-in "../foreign.rkt" cpu-device cuda-available? cuda-device
                    native-collect-at-troughs native-memory-unaccounted
                    native-memory-use)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/autograd-ops.rkt" reset-grads!)
           (only-in "../foreign/raw/device-queries.rkt" allocator-reading)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure.rkt"
                    native-collect-margin refresh-shadows! reset-pressure-state!)
           (only-in "../foreign/raw/race-points.rkt" race-hook)
           (only-in "../main.rkt"
                    * + backward! grad has-grad? ones reclaim-native-memory! sum
                    tensor tensor->list zeros)
           (only-in "../nn.rkt" Linear Parameter Sequential adam
                    clip-grad-norm! parameters rmsprop sgd step! step-lr
                    zero-grads!)
           (only-in "private/within.rkt" within))

  (define-syntax-parse-rule (bounded-test name:str body:expr ...+)
    (test-case name (within 60 (lambda () body ...))))

  (define (crossings-of thunk)
    (define here (current-thread))
    (define n 0)
    (with-crossing-hook (lambda () (when (eq? (current-thread) here) (set! n (add1 n))))
      (thunk))
    n)

  (define (released-during thunk)
    (define here (current-thread))
    (define n 0)
    (define previous (unbox race-hook))
    (dynamic-wind
     (lambda ()
       (set-box! race-hook
                 (lambda (name _subject)
                   (when (and (eq? name 'free-released) (eq? (current-thread) here))
                     (set! n (add1 n))))))
     thunk
     (lambda () (set-box! race-hook previous)))
    n)

  (define (backed-up width depth)
    (define net (apply Sequential (for/list ([_ (in-range depth)]) (Linear width width))))
    (backward! (sum (net (ones 2 width))))
    net)

  (bounded-test "zero-grads! crosses into native code once, whatever the parameter count"
    (for ([depth (in-list '(1 4 16))])
      (define net (backed-up 3 depth))
      (define opt (sgd (parameters net) #:lr 0.1))
      (check-equal? (length (parameters net)) (* 2 depth))
      (check-equal? (crossings-of (lambda () (zero-grads! opt))) 1
                    (format "~a parameters" (* 2 depth)))))

  (bounded-test "after zero-grads! no parameter has a gradient"
    (define net (backed-up 3 2))
    (define opt (adam (parameters net) #:lr 0.1))
    (check-true (andmap has-grad? (parameters net)))
    (zero-grads! opt)
    (check-false (ormap has-grad? (parameters net)))
    (zero-grads! opt)
    (check-false (ormap has-grad? (parameters net)) "a second reset is harmless"))

  (bounded-test "#:set-to-none? #f zeroes every gradient in place"
    (define net (backed-up 3 2))
    (define opt (sgd (parameters net) #:lr 0.1))
    (zero-grads! opt #:set-to-none? #f)
    (for ([p (in-list (parameters net))])
      (check-true (has-grad? p))
      (check-true (andmap zero? (tensor->list (grad p))))))

  (bounded-test "reset-grads! blames its caller for a list that is not of tensors"
    (check-exn exn:fail:contract:blame? (lambda () (reset-grads! '(1.0)))))

  (bounded-test "a backward after the reset gives a fresh backward's gradient"
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

  (bounded-test "a step after the reset leaves a parameter without a gradient alone"
    (define w (Parameter (tensor '(1.0 2.0))))
    (define opt (sgd (list w) #:lr 0.1 #:momentum 0.9))
    (backward! (sum w))
    (step! opt)
    (define stepped (tensor->list w))
    (zero-grads! opt)
    (step! opt)
    (check-equal? (tensor->list w) stepped "momentum moved a parameter with no gradient"))

  (bounded-test "zeroed in place, Adam's momentum moves a parameter its loss left out"
    (define a (Parameter (tensor '(1.0))))
    (define b (Parameter (tensor '(0.5))))
    (define opt (adam (list a b) #:lr 0.1))
    (for ([uses-b? (in-list '(#t #f))])
      (zero-grads! opt #:set-to-none? #f)
      (define loss (sum (* a a)))
      (backward! (if uses-b? (+ loss (sum (* b b))) loss))
      (step! opt))
    (check-= (car (tensor->list b)) 0.33299 1e-5))

  (bounded-test "Adam's bias correction counts the steps a parameter had a gradient for"
    (define a (Parameter (tensor '(1.0))))
    (define b (Parameter (tensor '(1.0))))
    (define opt (adam (list a b) #:lr 0.1))
    (backward! (sum a))
    (step! opt)
    (zero-grads! opt)
    (backward! (sum (* a b)))
    (step! opt)
    (check-= (car (tensor->list b)) 0.9 1e-6 "b's first step moves it by lr"))

  (bounded-test "a schedule drops its optimizer's gradients"
    (define w (Parameter (tensor '(1.0 2.0))))
    (define s (step-lr (sgd (list w) #:lr 0.1) #:step-size 1))
    (backward! (sum w))
    (zero-grads! s)
    (check-false (has-grad? w)))

  (bounded-test "a step releases every gradient handle it takes"
    (for ([make-optimizer
           (in-list (list (lambda (ps) (sgd ps #:lr 0.1 #:momentum 0.9 #:weight-decay 0.1))
                          (lambda (ps) (adam ps #:lr 0.1))
                          (lambda (ps) (rmsprop ps #:lr 0.1 #:momentum 0.5))))])
      (define ps (parameters (backed-up 3 2)))
      (define opt (make-optimizer ps))
      (check-equal? (released-during (lambda () (step! opt))) (length ps))))

  (bounded-test "clipping releases every gradient handle it takes"
    (define ps (parameters (backed-up 3 2)))
    (check-equal? (released-during (lambda () (clip-grad-norm! ps 0.1))) (length ps)))

  (bounded-test "zeroing in place releases every gradient handle it takes"
    (define net (backed-up 3 2))
    (define opt (sgd (parameters net) #:lr 0.1))
    (check-equal? (released-during (lambda () (zero-grads! opt #:set-to-none? #f)))
                  (length (parameters net))))

  (define mib (* 1024 1024))

  (define (cpu-bytes)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (cpu-unaccounted)
    (cond [(assoc (cpu-device) (native-memory-unaccounted)) => cdr]
          [else 0]))

  (define (settle!)
    (for ([_ (in-range 3)])
      (reclaim-native-memory!))
    (reset-pressure-state!))

  ;; the stand-in allocator charges the gradient's 4 MiB beyond the ledger
  ;; while only the parameter holds it
  (bounded-test "a gradient handle released after a reset leaves the ledger exact"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (define opt (sgd (list w) #:lr 0.1))
    (define unseen (box (* 4 mib)))
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (unbox unseen)))]
                   [native-collect-margin (* 1024 1024 mib)])
      (backward! (sum w))
      (check-equal? (cpu-unaccounted) (* 4 mib) "the trough charged the gradient")
      (define g (grad w))
      (set-box! unseen 0)
      (check-equal? (cpu-unaccounted) 0 "the handle adopted the gradient")
      (zero-grads! opt)
      (tensor-free! g)
      (define released (cpu-unaccounted))
      (refresh-shadows!)
      (check-equal? released (cpu-unaccounted)
                    "the release gave back bytes the parameter no longer holds"))
    (settle!))

  (when (cuda-available?)
    (bounded-test "on CUDA a step's gradients are gone before the next backward"
      (define cuda (cuda-device))
      (define numel (* 4 mib))
      (define nbytes (* 4 numel))
      (parameterize ([native-collect-at-troughs #f])
        (define w (Parameter (zeros numel #:device cuda)))
        (define opt (sgd (list w) #:lr 0.1))
        (define (gradients-held)
          (round (/ (- ((allocator-reading) cuda) base) nbytes)))
        (define base ((allocator-reading) cuda))
        (backward! (sum w))
        (step! opt)
        (zero-grads! opt)
        (backward! (sum w))
        (check-equal? (gradients-held) 1)))))
