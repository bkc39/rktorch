#lang racket/base

(module+ test
  (require (only-in racket/contract/combinator exn:fail:contract:blame?)
           rackunit
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/trainer-kit.rkt" toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "trainer-hparam-test"))

  (define-trainer Tuned (net lr width seen)
    (define-init ([b (box '())])
      (set! net (Linear 2 1))
      (set! lr 0.1)
      (set! width 4)
      (set! seen b))
    (define-hparams lr width)
    (define-optimizers (sgd (parameters net) #:lr lr))
    (define-training-step (x y)
      (set-box! seen (cons lr (unbox seen)))
      (mse-loss (net x) y)))

  (define-trainer Wordy (net lr)
    (define-init () (set! net (Linear 2 1)) (set! lr "fast"))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "G2-L5: set-hparam! then hparam answers the value, and so does the field"
    (define seen (box '()))
    (define tr (Tuned seen))
    (check-equal? (hparam tr 'lr) 0.1)
    (set-hparam! tr 'lr 0.5)
    (check-equal? (hparam tr 'lr) 0.5)
    (check-equal? (trainer-hparams tr) (hasheq 'lr 0.5 'width 4))
    (void (fit tr (toy-loader 4) #:max-epochs 1 #:logger #f))
    (check-equal? (unbox seen) '(0.5 0.5))
    (check-equal? (hparam tr 'lr) 0.5))

  (bounded-test "G2-L6: a non-real hparam after define-init names the field"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? #rx"an hparam must hold a real.*hparam: 'lr"
                                     (exn-message e))))
               (lambda () (Wordy))))

  (define-trainer Retyped (net lr)
    (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (set! lr "fast")
      (mse-loss (net x) y)))

  (define (names-the-field? who)
    (lambda (e)
      (and (exn:fail:contract? e)
           (not (exn:fail:contract:blame? e))
           (regexp-match? (pregexp (format "^~a: an hparam must hold a real.*hparam: 'lr" who))
                          (exn-message e)))))

  (bounded-test "G2-U1: a field a step set to a non-real is reported naming it, not as blame"
    (define tr (Retyped))
    (void (fit tr (toy-loader 2) #:max-epochs 1 #:logger #f))
    (check-exn (names-the-field? 'hparam) (lambda () (hparam tr 'lr)))
    (check-exn (names-the-field? 'trainer-hparams) (lambda () (trainer-hparams tr))))

  (bounded-test "G2-L6: set-hparam! of an unknown name blames the caller"
    (check-exn blames-this-test? (lambda () (set-hparam! (Tuned) 'depth 3)))))
