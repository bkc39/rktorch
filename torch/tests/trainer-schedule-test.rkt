#lang racket/base

(module+ test
  (require (only-in racket/math pi)
           (only-in racket/match match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" cat item manual-seed! randn reshape)
           (only-in "../nn.rkt"
                    Linear adamw cosine-annealing-lr exponential-lr linear-lr mse-loss
                    param-group param-group-ref parameters rmsprop sequential-lr sgd step-lr)
           "../train.rkt"
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/lightning-names.rkt" fixture)
           (only-in "private/python-env.rkt" check-training-twin python-torch-available? tol)
           (only-in "private/trainer-kit.rkt" toy-loader)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path twin "python/schedule_callbacks_twin.py")

  (define (curve t) (+ 1.0 (* 0.5 t)))

  (define-trainer Tempered (net temp sink)
    (define-init (s) (set! net (Linear 2 1)) (set! temp 0.0) (set! sink s))
    (define-hparams temp)
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (set-box! sink (cons (list 'train (run-epoch run) (run-global-step run) temp)
                           (unbox sink)))
      (mse-loss (net x) y))
    (define-validation-step (x y)
      (set-box! sink (cons (list 'val (run-epoch run) (run-global-step run) temp) (unbox sink)))
      (mse-loss (net x) y)))

  (define (tempered-events cb #:train train #:accumulate accumulate)
    (define sink (box '()))
    (fit (Tempered sink) (toy-loader (* 2 train)) #:val (toy-loader 2) #:max-epochs 2
         #:accumulate-grad-batches accumulate #:num-sanity-val-steps 1 #:logger #f
         #:callbacks (list cb))
    (reverse (unbox sink)))

  (define (check-no-eval-writes events label)
    (for/fold ([last-train 0.0]) ([e (in-list events)])
      (match e
        [(list 'train _ _ temp) temp]
        [(list 'val _ step temp)
         (check-equal? temp last-train (format "~a: a validation batch at step ~a" label step))
         last-train]))
    (void))

  (bounded-test "G2-L7: a 'step hparam schedule writes curve(global step) at every batch"
    (for* ([accumulate (in-range 1 4)] [train (in-range 1 6)])
      (define label (format "A ~a, b ~a" accumulate train))
      (define events (tempered-events (hparam-scheduler 'temp curve) #:train train
                                      #:accumulate accumulate))
      (for ([e (in-list events)])
        (match e
          [(list 'train _ step temp)
           (check-equal? temp (curve step) (format "~a, step ~a" label step))]
          [_ (void)]))
      (check-no-eval-writes events label)))

  (bounded-test "G2-L7: an 'epoch hparam schedule writes curve(epoch) at every epoch start"
    (define events (tempered-events (hparam-scheduler 'temp curve #:interval 'epoch)
                                    #:train 3 #:accumulate 1))
    (for ([e (in-list events)])
      (match e
        [(list 'train epoch _ temp) (check-equal? temp (curve epoch))]
        [_ (void)]))
    (check-no-eval-writes events "'epoch"))

  (define-trainer Grouped (net opt-box)
    (define-init (sink) (set! net (Linear 2 2)) (set! opt-box sink))
    (define-optimizers
      (match-define (list weight bias) (parameters net))
      (define opt (adamw (list (param-group (list weight) #:weight-decay 0.05)
                               (param-group (list bias) #:weight-decay 0.0))))
      (set-box! opt-box opt)
      opt)
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "G2-L8: a param-group schedule writes only the selected groups"
    (define opt-box (box #f))
    (define r (fit (Grouped opt-box) (toy-loader 6) #:max-epochs 1 #:logger #f
                   #:callbacks (list (param-group-scheduler 'weight-decay curve #:groups '(0)))))
    (define opt (unbox opt-box))
    (check-equal? (param-group-ref opt 0 'weight-decay) (curve (sub1 (run-global-step r))))
    (check-equal? (param-group-ref opt 1 'weight-decay) 0.0))

  (bounded-test "an 'epoch param-group schedule writes every group at each epoch start"
    (define opt-box (box #f))
    (fit (Grouped opt-box) (toy-loader 6) #:max-epochs 3 #:logger #f
         #:callbacks (list (param-group-scheduler 'weight-decay curve #:interval 'epoch)))
    (define opt (unbox opt-box))
    (check-equal? (list (param-group-ref opt 0 'weight-decay) (param-group-ref opt 1 'weight-decay))
                  (list (curve 2) (curve 2))))

  (bounded-test "a beta1 schedule takes any real curve and writes the group's beta1"
    (define opt-box (box #f))
    (fit (Grouped opt-box) (toy-loader 4) #:max-epochs 1 #:logger #f
         #:callbacks (list (param-group-scheduler 'beta1 (lambda (t) (- 0.9 (* 0.01 t))))))
    (check-equal? (param-group-ref (unbox opt-box) 1 'beta1) (- 0.9 (* 0.01 1))))

  (define-trainer Momentum (net opt-box)
    (define-init (sink) (set! net (Linear 2 2)) (set! opt-box sink))
    (define-optimizers
      (match-define (list weight bias) (parameters net))
      (define opt (sgd (list (param-group (list weight)) (param-group (list bias) #:momentum 0.5))
                       #:lr 0.1 #:momentum 0.9))
      (set-box! opt-box opt)
      opt)
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "a momentum schedule writes every SGD group's momentum"
    (define opt-box (box #f))
    (define (momentum t) (- 0.9 (* 0.01 t)))
    (define r (fit (Momentum opt-box) (toy-loader 6) #:max-epochs 1 #:logger #f
                   #:callbacks (list (param-group-scheduler 'momentum momentum))))
    (define opt (unbox opt-box))
    (define want (momentum (sub1 (run-global-step r))))
    (check-equal? (list (param-group-ref opt 0 'momentum) (param-group-ref opt 1 'momentum))
                  (list want want)))

  (define-trainer Smoothed (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (rmsprop (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "a curve value the optimizer rejects raises naming param-group-scheduler"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? #rx"^param-group-scheduler: " (exn-message e))
                      (not (regexp-match? #rx"blaming: [^\n]*schedules[.]rkt" (exn-message e)))))
               (lambda ()
                 (fit (Smoothed) (toy-loader 4) #:max-epochs 1 #:logger #f
                      #:callbacks (list (param-group-scheduler 'eps (lambda (_t) -1.0)))))))

  (define start-errors
    `(("an unknown hparam" hparam-scheduler ,(hparam-scheduler 'depth curve))
      ("an optimizer index past the last" param-group-scheduler
       ,(param-group-scheduler 'lr curve #:optimizer 1))
      ("a group index past the last" param-group-scheduler
       ,(param-group-scheduler 'lr curve #:groups '(2)))
      ("a key the optimizer lacks" param-group-scheduler
       ,(param-group-scheduler 'momentum curve))))

  (for ([case (in-list start-errors)])
    (match-define (list label name cb) case)
    (bounded-test (format "G2-L9: ~a raises at train start, naming the callback" label)
      (check-exn (lambda (e)
                   (and (exn:fail:contract? e)
                        (regexp-match? (regexp (symbol->string name)) (exn-message e))))
                 (lambda ()
                   (fit (Grouped (box #f)) (toy-loader 4) #:max-epochs 1 #:logger #f
                        #:callbacks (list cb))))))

  (define-trainer Scheduled (net seen)
    (define-init (sink) (set! net (Linear 2 1)) (set! seen sink))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1))
      (define per-step (scheduler-config (step-lr opt #:step-size 2) #:interval 'step
                                         #:frequency 2))
      (define per-epoch (exponential-lr opt #:gamma 0.9))
      (set-box! seen (list per-step per-epoch (run-estimated-stepping-batches run)))
      (values opt per-step per-epoch))
    (define-training-step (x y) (mse-loss (net x) y)))

  (bounded-test "G2-L10: run-schedulers answers define-optimizers' configs in order"
    (define seen (box #f))
    (define r (fit (Scheduled seen) (toy-loader 4) #:max-epochs 1 #:logger #f))
    (match-define (list per-step per-epoch _) (unbox seen))
    (match-define (list a b) (run-schedulers r))
    (check-eq? a per-step)
    (check-eq? (scheduler-config-scheduler b) per-epoch)
    (check-equal? (map scheduler-config-interval (list a b)) '(step epoch))
    (check-equal? (map scheduler-config-frequency (list a b)) '(2 1)))

  (define-trainer Evaluated (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1))
      (values opt (exponential-lr opt #:gamma 0.9)))
    (define-test-step (x y) (mse-loss (net x) y))
    (define-predict-step (_x _y) (run-schedulers run)))

  (bounded-test "a test or predict run has no schedulers and no stepping estimate"
    (define r (test (Evaluated) (toy-loader 2) #:logger #f))
    (check-equal? (run-schedulers r) '())
    (check-equal? (predict (Evaluated) (toy-loader 2)) '(()))
    (check-exn (blames-caller? "trainer-schedule-test")
               (lambda () (run-estimated-stepping-batches r))))

  (bounded-test "G2-L11: run-estimated-stepping-batches is Lightning's for the fixture table"
    (for ([row (in-list (hash-ref fixture 'stepping))])
      (define seen (box #f))
      (define (opt key) (hash-ref row key #f))
      (fit (Scheduled seen) (toy-loader (* 2 (hash-ref row 'train))) #:logger #f
           #:max-epochs (opt 'max-epochs) #:max-steps (opt 'max-steps)
           #:accumulate-grad-batches (hash-ref row 'accumulate)
           #:limit-train-batches (hash-ref row 'limit)
           #:fast-dev-run (opt 'fast-dev-run))
      (match-define (list _ _ estimated) (unbox seen))
      (check-equal? estimated (hash-ref row 'estimated) (format "~s" row))))

  (bounded-test "with neither limit fit refuses before define-optimizers reads an estimate"
    (define seen (box #f))
    (check-exn (blames-caller? "trainer-schedule-test")
               (lambda () (fit (Scheduled seen) (toy-loader 4) #:logger #f
                               #:max-epochs #f #:max-steps #f)))
    (check-false (unbox seen)))

  (define (warm-then-cosine opt)
    (sequential-lr opt (list (linear-lr opt #:start-factor 0.1 #:total-iters 3)
                             (cosine-annealing-lr opt #:t-max 5 #:eta-min 1e-4))
                   #:milestones '(3)))

  (define (dino-wd t) (+ 0.04 (* 0.1 (- 1.0 (cos (/ (* pi t) 8))))))

  (define-trainer Dino (net losses)
    (define-init (model sink) (set! net model) (set! losses sink))
    (define-optimizers
      (match-define (list weight bias) (parameters net))
      (define opt (adamw (list (param-group (list weight) #:weight-decay 0.04)
                               (param-group (list bias) #:weight-decay 0.0))
                         #:lr 1e-2))
      (values opt (scheduler-config (warm-then-cosine opt) #:interval 'step)))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (set-box! losses (cons (item loss) (unbox losses)))
      loss))

  (define (fit-on device)
    (manual-seed! 0)
    (define net (Linear 4 2))
    (define xs (randn 16 4))
    (define ys (randn 16 2))
    (define losses (box '()))
    (fit (Dino net losses) (dataloader (tensor-dataset xs ys) #:batch-size 4)
         #:max-epochs 2 #:logger #f
         #:callbacks (list (param-group-scheduler 'weight-decay dino-wd #:groups '(0))))
    (values (reverse (unbox losses))
            (cat (for/list ([p (in-list (parameters net))]) (reshape p -1)))))

  (bounded-test "G2-L12: a DINO-style schedule run against the plain torch loop"
    (cond
      [(not (python-torch-available?))
       (displayln "[trainer-schedule-test] G2-L12 skipped: python3 `torch` not available")]
      [else (check-training-twin "schedule callbacks" twin fit-on 'cpu tol)])))
