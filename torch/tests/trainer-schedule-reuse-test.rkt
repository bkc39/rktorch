#lang racket/base

(module+ test
  (require (only-in racket/match match match-define)
           rackunit
           (only-in "../nn.rkt" Linear adamw mse-loss param-group-ref parameters sgd)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt" toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define gate-seconds 30)

  (define (curve t) (+ 100.0 t))

  (define-trainer Gated (net temp sink gate)
    (define-init (s g) (set! net (Linear 2 1)) (set! temp 0.0) (set! sink s) (set! gate g))
    (define-hparams temp)
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (when (zero? (run-global-step run)) (gate))
      (set-box! sink (cons (list (run-global-step run) temp) (unbox sink)))
      (mse-loss (net x) y)))

  (define (await sema who)
    (unless (sync/timeout gate-seconds sema)
      (error 'trainer-schedule-reuse-test "~a was not reached within ~a s" who gate-seconds)))

  (define (fit-gated cb gate)
    (define sink (box '()))
    (fit (Gated sink gate) (toy-loader 8) #:max-epochs 1 #:logger #f #:callbacks (list cb))
    (reverse (unbox sink)))

  (define (fit-in-thread thunk)
    (define result (box #f))
    (define t
      (thread (lambda ()
                (set-box! result (with-handlers ([exn? values]) (list (thunk)))))))
    (lambda ()
      (unless (sync/timeout gate-seconds t)
        (kill-thread t)
        (error 'trainer-schedule-reuse-test "the first fit did not finish within ~a s"
               gate-seconds))
      (match (unbox result)
        [(list v) v]
        [e (raise e)])))

  (define (check-own-curve events label)
    (for ([e (in-list events)])
      (match-define (list step temp) e)
      (check-equal? temp (curve step) (format "~a, step ~a" label step))))

  (bounded-test "one hparam-scheduler shared by two concurrent fits writes each its own curve"
    (define cb (hparam-scheduler 'temp curve))
    (define a-at-step-0 (make-semaphore 0))
    (define b-started (make-semaphore 0))
    (define a-events
      (fit-in-thread
       (lambda ()
         (fit-gated cb (lambda ()
                         (semaphore-post a-at-step-0)
                         (await b-started "the second fit's first step"))))))
    (await a-at-step-0 "the first fit's first step")
    (define b-events (fit-gated cb (lambda () (semaphore-post b-started))))
    (define a (a-events))
    (check-own-curve a "the first fit")
    (check-equal? (length a) 4)
    (check-own-curve b-events "the second fit")
    (check-equal? (length b-events) 4))

  (define-trainer Decayed (net opt-box)
    (define-init (o) (set! net (Linear 2 2)) (set! opt-box o))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1 #:weight-decay 0.0))
      (set-box! opt-box opt)
      opt)
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "one param-group-scheduler reused by two fits writes the second fit's optimizer"
    (define cb (param-group-scheduler 'weight-decay (lambda (t) (+ 1.0 t))))
    (define (fit-once)
      (define opt-box (box #f))
      (fit (Decayed opt-box) (toy-loader 6) #:max-epochs 1 #:logger #f #:callbacks (list cb))
      (param-group-ref (unbox opt-box) 0 'weight-decay))
    (check-equal? (list (fit-once) (fit-once)) (list 3.0 3.0)))

  (define-trainer Retained (net wb)
    (define-init (w) (set! net (Linear 2 2)) (set! wb w))
    (define-optimizers
      (define opt (adamw (parameters net) #:lr 1e-3))
      (set-box! wb (make-weak-box opt))
      opt)
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "a param-group-scheduler kept after its fit does not keep the optimizer alive"
    (define cb (param-group-scheduler 'weight-decay (lambda (t) (+ 1.0 t))))
    (define wb (box #f))
    (fit (Retained wb) (toy-loader 6) #:max-epochs 1 #:logger #f #:callbacks (list cb))
    (for ([_ (in-range 3)]) (collect-garbage 'major))
    (check-false (weak-box-value (unbox wb)))
    (check-pred callback? cb)))
