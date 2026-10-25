#lang racket/base

(module+ test
  (require rackunit
           (only-in ffi/unsafe ptr-add set-cpointer-tag!)
           (only-in ffi/unsafe/atomic in-atomic-mode?)
           (only-in racket/contract exn:fail:contract:blame?)
           (only-in racket/list make-list)
           (only-in "../foreign.rkt" cpu-device cuda-available? cuda-device
                    native-collect-at-troughs native-memory-unaccounted
                    native-memory-use)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/autograd-ops.rkt" reset-grads!)
           (only-in "../foreign/raw/device-queries.rkt" allocator-reading)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/memory.rkt" adopted-bytes)
           (only-in "../foreign/raw/pressure.rkt"
                    native-collect-margin refresh-shadows!)
           (only-in "../main.rkt"
                    * + backward! grad has-grad? ones sum tensor tensor->list
                    zeros)
           (only-in "../nn.rkt" Linear Parameter Sequential adam
                    clip-grad-norm! parameters rmsprop sgd step! step-lr
                    zero-grads!)
           (only-in "../foreign/structs.rkt" tensor-handle tensor-impl wrap-tensor)
           (only-in "private/race-harness.rkt" hooks settle! with-race-hook)
           (only-in "private/within.rkt" bounded-test))

  (define (crossings-of thunk)
    (define here (current-thread))
    (define n 0)
    (with-crossing-hook (lambda () (when (eq? (current-thread) here) (set! n (add1 n))))
      (thunk))
    n)

  (define (no-hook _name _subject)
    (void))

  (define (released-during thunk #:also [also no-hook])
    (define here (current-thread))
    (define n 0)
    (define (count! name _subject)
      (when (and (eq? name 'free-released) (eq? (current-thread) here))
        (set! n (add1 n))))
    (with-race-hook (hooks count! also)
      (thunk))
    n)

  (define ((on-nth point n action) name _subject)
    (when (eq? name point)
      (set! n (sub1 n))
      (when (zero? n)
        (action))))

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

  (define (sgd-with-momentum ps)
    (sgd ps #:lr 0.1 #:momentum 0.9 #:weight-decay 0.1))

  (define (plain-adam ps)
    (adam ps #:lr 0.1))

  (define (rmsprop-with-momentum ps)
    (rmsprop ps #:lr 0.1 #:momentum 0.5))

  (bounded-test "a step releases every gradient handle it takes"
    (for ([make-optimizer (in-list (list sgd-with-momentum plain-adam rmsprop-with-momentum))])
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

  (bounded-test "a raise on the third gradient fetch releases the two already taken"
    (define ps (parameters (backed-up 3 2)))
    (define (raise-injected) (raise 'injected))
    (check-equal? (released-during
                   (lambda ()
                     (with-handlers ([(lambda (e) (eq? e 'injected)) void])
                       (clip-grad-norm! ps 0.1)))
                   #:also (on-nth 'op-returned 3 raise-injected))
                  2))

  (define mib (* 1024 1024))

  (define (cpu-entry-of readings)
    (cond [(assoc (cpu-device) readings) => cdr]
          [else 0]))

  (define (cpu-bytes)
    (cpu-entry-of (native-memory-use)))

  (define (cpu-unaccounted)
    (cpu-entry-of (native-memory-unaccounted)))

  (define (with-unseen-gradient unseen thunk)
    (parameterize ([allocator-reading (lambda (_dev) (+ (cpu-bytes) (unbox unseen)))]
                   [native-collect-margin (* 1024 1024 mib)])
      (thunk)))

  (bounded-test "a gradient handle released after a reset leaves the ledger exact"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (define opt (sgd (list w) #:lr 0.1))
    (define unseen (box (* 4 mib)))
    (with-unseen-gradient unseen
      (lambda ()
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
                      "the release gave back bytes the parameter no longer holds")))
    (settle!))

  (define (adopted g)
    (adopted-bytes (tensor-handle g)))

  (define (adopting-gradient thunk)
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (define unseen (box (* 4 mib)))
    (with-unseen-gradient unseen
      (lambda ()
        (backward! (sum w))
        (define g (grad w))
        (set-box! unseen 0)
        (thunk w g)))
    (settle!))

  (bounded-test "a reset forgets the adoptions in atomic mode, after the native reset succeeded"
    (adopting-gradient
     (lambda (w g)
       (define at-forget #f)
       (define (observe! name _ts)
         (when (eq? name 'adoptions-forgotten)
           (set! at-forget (list (in-atomic-mode?) (has-grad? w) (adopted g)))))
       (with-race-hook observe!
         (zero-grads! (sgd (list w) #:lr 0.1)))
       (check-equal? at-forget (list #t #f (* 4 mib))
                     "in atomic mode, after the native reset succeeded, the adoption still held")
       (check-equal? (adopted g) 0 "the reset forgot the adoption")
       (tensor-free! g))))

  (bounded-test "a raise while adopting a gradient propagates, and the handle is released once"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (backward! (sum w))
    (settle!)
    (define before (cpu-bytes))
    (define taken #f)
    (define (raise-adopting name t)
      (when (eq? name 'gradient-adopting)
        (set! taken t)
        (raise (exn:fail "injected adoption failure" (current-continuation-marks)))))
    (check-exn #rx"injected adoption failure"
               (lambda () (with-race-hook raise-adopting (grad w))))
    (check-false (in-atomic-mode?) "the raise left atomic mode")
    (check-equal? (cpu-bytes) (+ before (* 4 mib)) "the handle is on the ledger")
    (check-equal? (released-during (lambda () (tensor-free! (wrap-tensor taken)))) 1)
    (check-equal? (cpu-bytes) before "the release took the handle off the ledger")
    (settle!)
    (check-equal? (cpu-bytes) before "no finalizer released it again")
    (check-true (has-grad? w)))

  (define null-tensor
    (let ([h (ptr-add #f 0)])
      (set-cpointer-tag! h 'Tensor)
      (tensor-impl h '())))

  (bounded-test "a reset native code refuses keeps the adoptions"
    (adopting-gradient
     (lambda (w g)
       (check-exn #rx"zero-grads!" (lambda () (reset-grads! (list w null-tensor))))
       (check-true (has-grad? w))
       (check-equal? (adopted g) (* 4 mib))
       (tensor-free! g)
       (check-equal? (cpu-unaccounted) (* 4 mib) "the release gave the adoption back"))))

  (define (handles-across-a-refresh k reset?)
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (define opt (sgd (list w) #:lr 0.1))
    (define gradient (* 4 mib))
    (define other (* 8 mib))
    (define unseen (box (+ other gradient)))
    (with-unseen-gradient unseen
      (lambda ()
        (backward! (sum w))
        (define handles
          (for/list ([i (in-range k)])
            (refresh-shadows!)
            (define h (grad w))
            (set-box! unseen (- other (* i gradient)))
            h))
        (check-equal? (map adopted handles) (cons gradient (make-list (sub1 k) 0))
                      "only the first handle adopts, even after a refresh")
        (refresh-shadows!)
        (define expected
          (cond
            [reset? (zero-grads! opt) other]
            [else (+ other gradient)]))
        (for-each tensor-free! handles)
        (set-box! unseen expected)
        (check-equal? (cpu-unaccounted) (- expected (* (sub1 k) gradient))
                      "the releases leave the ledger under by the extra handles' charges")
        (refresh-shadows!)
        (check-equal? (cpu-unaccounted) expected "the next refresh restores it")))
    (settle!))

  (bounded-test "two handles across a refresh, then a reset: one gradient under until a refresh"
    (handles-across-a-refresh 2 #t))

  (bounded-test "two handles across a refresh, no reset: one gradient under until a refresh"
    (handles-across-a-refresh 2 #f))

  (bounded-test "three handles across a refresh: two gradients under until a refresh"
    (handles-across-a-refresh 3 #f))

  (bounded-test "a refresh right after a gradient handle is accounted finds it adopted"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (define other (* 8 mib))
    (define unseen (box (+ other (* 4 mib))))
    (with-unseen-gradient unseen
      (lambda ()
        (backward! (sum w))
        (define (refresh-in-the-gap!)
          (set-box! unseen other)
          (refresh-shadows!))
        (define g
          (with-race-hook (on-nth 'tensor-accounted 1 refresh-in-the-gap!)
            (grad w)))
        (check-equal? (cpu-unaccounted) other "the gradient is charged once, on the ledger")
        (tensor-free! g)))
    (settle!))

  (bounded-test "after clipping, a step's gradient handle is charged once"
    (settle!)
    (define w (zeros 1024 1024 #:requires-grad? #t))
    (define unseen (box (* 4 mib)))
    (with-unseen-gradient unseen
      (lambda ()
        (backward! (sum w))
        (clip-grad-norm! (list w) 1e9)
        (check-equal? (cpu-unaccounted) (* 4 mib) "the clip's release gave the charge back")
        (define g (grad w))
        (set-box! unseen 0)
        (check-equal? (cpu-unaccounted) 0 "the step's handle took the charge over")
        (tensor-free! g)))
    (settle!))

  (define (gradients-gone-on-cuda)
    (define cuda (cuda-device))
    (define numel (* 4 mib))
    (define nbytes (* 4 numel))
    (parameterize ([native-collect-at-troughs #f])
      (define w (Parameter (zeros numel #:device cuda)))
      (define opt (sgd (list w) #:lr 0.1))
      (define base ((allocator-reading) cuda))
      (define (gradients-held)
        (round (/ (- ((allocator-reading) cuda) base) nbytes)))
      (backward! (sum w))
      (step! opt)
      (zero-grads! opt)
      (backward! (sum w))
      (check-equal? (gradients-held) 1)))

  (bounded-test "on CUDA a step's gradients are gone before the next backward"
    (cond
      [(cuda-available?) (gradients-gone-on-cuda)]
      [else (displayln "[zero-grads-test] skipped: no CUDA device")])))
