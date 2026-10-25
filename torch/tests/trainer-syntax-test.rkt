#lang racket/base

(module trainers racket/base
  (require (only-in racket/contract/base -> any/c)
           (only-in "../train.rkt" define-init define-trainer))

  (define-trainer Classifier (net)
    #:contract (-> any/c classifier?)
    (define-init (n) (set! net n))))

(module+ test
  (require rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           (submod ".." trainers)
           "../train.rkt")

  (define ((message-matching pattern) e)
    (and (exn:fail:syntax? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (test-case "E1: an unknown hook names the closest one"
    (check-exn
     (message-matching
      #rx"define-hook: unknown hook on-batch-end; closest: on-train-batch-end")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Typo () (define-hook (on-batch-end) (void)))
          Typo))))
    (check-exn
     (message-matching #rx"define-hook: unknown hook on-fit-strat; closest: on-fit-start")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback typo () (define-hook (on-fit-strat) (void)))
          typo)))))

  (test-case "E2: a hook defined on the wrong side"
    (check-exn
     (message-matching
      #rx"define-hook: on-after-batch-transfer is a trainer hook; define-callback cannot define it")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback moved ()
            (define-hook (on-after-batch-transfer . batch) (apply values batch)))
          moved))))
    (check-exn
     (message-matching
      #rx"define-hook: on-exception is a callback hook; define-trainer cannot define it")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Catching () (define-hook (on-exception e) (void)))
          Catching)))))

  (test-case "E3: a hook's arity"
    (check-exn
     (message-matching
      #rx"define-hook: on-before-optimizer-step takes 1 argument [(]optimizer[)]")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Stepper () (define-hook (on-before-optimizer-step) (void)))
          Stepper))))
    (check-exn
     (message-matching
      #rx"define-hook: on-train-batch-end takes 2 arguments [(]loss extra[)] then the batch values")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Ender () (define-hook (on-train-batch-end loss) (void)))
          Ender))))
    (check-exn
     (message-matching #rx"define-hook: on-fit-start takes no arguments")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback starter () (define-hook (on-fit-start . more) (void)))
          starter))))
    (check-exn
     (message-matching #rx"define-hook: setup takes 1 argument [(]stage[)]")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback setter () (define-hook (setup stage extra) (void)))
          setter)))))

  (test-case "E4: a body form or a hook given twice"
    (check-exn
     (message-matching #rx"define-trainer: define-optimizers appears twice")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Twice ()
            (define-optimizers 1)
            (define-optimizers 2))
          Twice))))
    (check-exn
     (message-matching #rx"define-hook: on-fit-start is defined twice")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback twice ()
            (define-hook (on-fit-start) 1)
            (define-hook (on-fit-start) 2))
          twice)))))

  (test-case "E5: a body form of the other definer"
    (check-exn
     (message-matching
      #rx"define-callback: define-training-step is a define-trainer form")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback stepping () (define-training-step (x) x))
          stepping))))
    (check-exn
     (message-matching
      #rx"define-callback: define-optimizers is a define-trainer form")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback optimizing () (define-optimizers 1))
          optimizing))))
    (check-exn
     (message-matching
      #rx"define-callback: define-callbacks is a define-trainer form")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback nesting () (define-callbacks 1))
          nesting))))
    (check-exn
     (message-matching
      #rx"define-callback: define-hparams is a define-trainer form")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback tuned (lr) (define-hparams lr))
          tuned))))
    (check-exn
     (message-matching
      #rx"define-callback: define-forward is a define-trainer form")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback forwarding () (define-forward (x) x))
          forwarding))))
    (check-exn
     (message-matching
      #rx"define-trainer: define-state-key is a define-callback form")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Keyed () (define-state-key "keyed"))
          Keyed)))))

  (test-case "E6: run and log! outside a hook body"
    (check-exn
     (message-matching
      #rx"log!: only allowed inside a trainer or callback hook body")
     (lambda () (convert-compile-time-error (log! 'loss 1.0))))
    (check-exn
     (message-matching
      #rx"log!: only allowed inside a trainer or callback hook body")
     (lambda () (convert-compile-time-error log!)))
    (check-exn
     (message-matching
      #rx"run: only allowed inside a trainer or callback hook body")
     (lambda () (convert-compile-time-error (run-epoch run)))))

  (test-case "E8: define-hparams names fields"
    (check-exn
     (message-matching #rx"define-hparams: lr is not a field of Classifier")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Classifier (net momentum)
            (define-hparams momentum lr))
          Classifier)))))

  (test-case "E9: a body form outside a definer"
    (check-exn
     (message-matching
      #rx"define-init: only allowed directly inside define-trainer or define-callback")
     (lambda () (convert-compile-time-error (define-init () (void)))))
    (check-exn
     (message-matching
      #rx"define-hook: only allowed directly inside define-trainer or define-callback")
     (lambda ()
       (convert-compile-time-error (define-hook (on-fit-start) (void)))))
    (check-exn
     (message-matching
      #rx"define-training-step: only allowed directly inside define-trainer or define-callback")
     (lambda ()
       (convert-compile-time-error (define-training-step (x) x)))))

  (test-case "E10: fields follow define-layer's rules"
    (check-exn
     (message-matching
      #rx"define-trainer: with define-init, a field is a bare identifier")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-trainer Defaulted ([lr 0.1]) (define-init () (void)))
          Defaulted))))
    (check-exn
     (message-matching #rx"a field name may not contain a dot")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback dotted (ema.decay))
          dotted)))))

  (test-case "E11: a callback's kind"
    (check-exn
     (message-matching
      #rx"define-callback: expected callback kind [(]plain, monitor or checkpoint[)]")
     (lambda ()
       (convert-compile-time-error
        (let ()
          (define-callback loud () #:kind loud)
          loud)))))

  (test-case "a definition that passes the checks is a T1 stub"
    (define-trainer Ender (net)
      #:automatic-optimization? #f
      (define-init (n) (set! net n))
      (define-forward (x) (net x))
      (define-hparams net)
      (define-training-step (x [y : real?]) (net x y))
      (define-hook (on-train-batch-end loss extra x y) (void))
      (define-hook (on-train-batch-start . batch) (void)))
    (define-callback watcher (seen)
      #:kind monitor
      (define-state-key "watcher")
      (define-hook (on-exception e) (set! seen e)))
    (check-false (Ender? 'anything))
    (check-exn #rx"define-trainer: not implemented until T1 [(]#384[)]"
               (lambda () (Ender 1)))
    (check-exn exn:fail:unsupported? (lambda () (Ender 1)))
    (check-exn #rx"define-callback: not implemented until T1 [(]#384[)]"
               (lambda () (watcher 1)))
    (check-exn #rx"define-trainer: not implemented until T1 [(]#384[)]"
               (lambda () (Classifier 1)))
    (check-false (classifier? 'anything))))
