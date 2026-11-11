#lang racket/base

(module+ test
  (require (only-in racket/list remove-duplicates)
           rackunit
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/trainer-kit.rkt" toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "trainer-accumulation-test"))

  (define-trainer Plain (net factors)
    (define-init (sink) (set! net (Linear 2 1)) (set! factors sink))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (set-box! factors (cons (list (run-epoch run) (run-accumulate-grad-batches run))
                              (unbox factors)))
      (mse-loss (net x) y)))

  (define-trainer Manual (net)
    #:automatic-optimization? #f
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) #f))

  (define (ceiling-quotient n d) (quotient (+ n d -1) d))

  (bounded-test "G3-L17: the scheduler sets each epoch's factor, and the step count follows"
    (define factors (box '()))
    (define r (fit (Plain factors) (toy-loader 20) #:max-epochs 2 #:logger #f
                   #:callbacks (list (gradient-accumulation-scheduler (hasheqv 1 2)))))
    (check-equal? (remove-duplicates (reverse (unbox factors))) '((0 1) (1 2)))
    (check-equal? (run-global-step r) (+ (ceiling-quotient 10 1) (ceiling-quotient 10 2))))

  (bounded-test "G3-L17: an odd batch count ends each epoch's last window short"
    (define r (fit (Plain (box '())) (toy-loader 10) #:max-epochs 3 #:logger #f
                   #:callbacks (list (gradient-accumulation-scheduler (hasheqv 0 2 2 3)))))
    (check-equal? (run-global-step r)
                  (+ (ceiling-quotient 5 2) (ceiling-quotient 5 2) (ceiling-quotient 5 3))))

  (bounded-test "G3-L18: the scheduler with #:accumulate-grad-batches 2 raises at train start"
    (check-exn (lambda (e) (and (exn:fail:contract? e)
                                (regexp-match? #rx"accumulate" (exn-message e))))
               (lambda ()
                 (fit (Plain (box '())) (toy-loader 4) #:max-epochs 1 #:logger #f
                      #:accumulate-grad-batches 2
                      #:callbacks (list (gradient-accumulation-scheduler (hasheqv 0 2)))))))

  (bounded-test "G3-L18: the scheduler under manual optimization raises at train start"
    (check-exn (lambda (e) (and (exn:fail:contract? e)
                                (regexp-match? #rx"manual optimization" (exn-message e))))
               (lambda ()
                 (fit (Manual) (toy-loader 4) #:max-epochs 1 #:logger #f
                      #:callbacks (list (gradient-accumulation-scheduler (hasheqv 0 2)))))))

  (bounded-test "an empty, mutable or zero-factor scheduling blames the caller"
    (for ([bad (list (hasheqv) (make-hasheqv '((0 . 2))) (hasheqv 0 0))])
      (check-exn blames-this-test? (lambda () (gradient-accumulation-scheduler bad))))))
