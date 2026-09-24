#lang racket/base

;; CPU cases run everywhere with bfloat16, the only half dtype CPU autocast
;; casts to on every libtorch build; the CUDA case is `when`-guarded.

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt")

  (test-case "autocast is off by default and reports the device's dtype"
    (check-false (autocast-enabled? 'cpu))
    (check-false (autocast-enabled? (cpu-device)))
    (check-equal? (autocast-dtype 'cpu) 'bfloat16))

  (test-case "with-autocast casts matmul to bfloat16 and restores on exit"
    (define a (tensor '((1.0 2.0) (3.0 4.0))))
    (define b (tensor '((0.5 0.0) (0.0 0.5))))
    (define inside
      (with-autocast #:device 'cpu
        (check-true (autocast-enabled? 'cpu))
        (check-equal? (autocast-dtype 'cpu) 'bfloat16)
        (matmul a b)))
    (check-equal? (tensor-dtype inside) 'bfloat16)
    (check-equal? (tensor->list inside) '(0.5 1.0 1.5 2.0))
    (check-false (autocast-enabled? 'cpu))
    (check-equal? (tensor-dtype (matmul a b)) 'float32)
    (check-equal? (tensor-dtype a) 'float32 "the inputs are never touched"))

  (test-case "leaving restores the dtype even where autocast was off"
    (check-false (autocast-enabled? 'cpu))
    (define before (autocast-dtype 'cpu))
    (define other (if (eq? before 'float16) 'bfloat16 'float16))
    (with-autocast #:device 'cpu #:dtype other
      (check-equal? (autocast-dtype 'cpu) other))
    (check-false (autocast-enabled? 'cpu))
    (check-equal? (autocast-dtype 'cpu) before
                  "the body's dtype does not outlive the extent"))

  (test-case "with-autocast nests and restores the outer state, even on escape"
    (with-autocast #:device 'cpu #:dtype 'bfloat16
      (with-autocast #:device 'cpu #:dtype 'bfloat16
        (check-true (autocast-enabled? 'cpu)))
      (check-true (autocast-enabled? 'cpu) "the outer extent is still on"))
    (check-false (autocast-enabled? 'cpu))
    (check-exn #rx"escaped"
               (lambda ()
                 (with-autocast #:device 'cpu
                   (error 'test "escaped"))))
    (check-false (autocast-enabled? 'cpu) "an exception still restores"))

  (test-case "call-with-autocast takes a device struct and a thunk"
    (define out
      (call-with-autocast (lambda () (matmul (ones 2 2) (ones 2 2)))
                          #:device (cpu-device)))
    (check-equal? (tensor-dtype out) 'bfloat16)
    (check-equal? (tensor->list out) '(2.0 2.0 2.0 2.0)))

  (test-case "with-autocast rejects a full-precision dtype as contract blame"
    (check-exn exn:fail:contract?
               (lambda () (with-autocast #:device 'cpu #:dtype 'float32 1)))
    (check-exn exn:fail:contract?
               (lambda () (call-with-autocast (lambda () 1) #:dtype 'int64))))

  (test-case "a Linear trains under autocast: half forward, float32 grads"
    (manual-seed! 0)
    (define l (Linear 4 2))
    (define x (randn 8 4))
    (define loss
      (with-autocast #:device 'cpu
        (define y (l x))
        (check-equal? (tensor-dtype y) 'bfloat16)
        (mean (mul y y))))
    (backward! loss)
    (for ([p (in-list (parameters l))])
      (check-true (has-grad? p))
      (check-equal? (tensor-dtype (grad p)) 'float32)
      (check-equal? (tensor-dtype p) 'float32 "the weights stay float32")))

  (test-case "autocast on CUDA casts a convolution"
    (when (cuda-available?)
      (define conv (to (Conv2d 3 8 3) (cuda-device)))
      (define x (to (randn 2 3 8 8) (cuda-device)))
      (check-false (autocast-enabled? 'cuda))
      (define y (with-autocast #:device 'cuda (conv x)))
      (check-equal? (tensor-dtype y) 'bfloat16)
      (check-equal? (tensor-device y) (cuda-device))
      (define z (with-autocast #:device 'cuda #:dtype 'float16 (conv x)))
      (check-equal? (tensor-dtype z) 'float16)
      (check-false (autocast-enabled? 'cuda))
      (check-equal? (tensor-dtype (conv x)) 'float32)))

  (test-case "a nested extent in the other half dtype does not reuse casts"
    (when (cuda-available?)
      (define lin (to (Linear 8 4) (cuda-device)))
      (define x (to (randn 2 8) (cuda-device)))
      (with-autocast #:device 'cuda #:dtype 'bfloat16
        (check-equal? (tensor-dtype (lin x)) 'bfloat16)
        (with-autocast #:device 'cuda #:dtype 'float16
          (check-equal? (tensor-dtype (lin x)) 'float16
                        "the outer extent's bfloat16 weight cast is not reused"))
        (check-equal? (tensor-dtype (lin x)) 'bfloat16)))))
