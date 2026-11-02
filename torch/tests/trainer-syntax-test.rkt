#lang racket/base

(module trainers racket/base
  (require (only-in racket/contract/base -> any/c)
           (only-in "../train.rkt" define-init define-trainer))

  (define-trainer Classifier (net)
    #:contract (-> any/c classifier?)
    (define-init (n) (set! net n))))

(module+ test
  (require (for-syntax racket/base (only-in syntax/parse/pre expr))
           (only-in racket/stxparam syntax-parameterize)
           rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           (only-in syntax/parse/define define-syntax-parse-rule)
           (submod ".." trainers)
           (submod "../train/trainer.rkt" private)
           "../train.rkt"
           (only-in "../train.rkt" [define-hook hk] [define-hparams hp] [define-optimizers opt]))

  (define ((message-matching pattern) e)
    (and (exn:fail:syntax? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (define-syntax-parse-rule (check-syntax-error pattern:expr form:expr)
    (check-exn (message-matching pattern)
               (lambda () (convert-compile-time-error form))))

  (test-case "E1: an unknown hook names the closest one"
    (check-syntax-error
     #rx"define-hook: unknown hook on-batch-end; closest: on-train-batch-end"
     (let ()
       (define-trainer Typo () (define-hook (on-batch-end) (void)))
       Typo))
    (check-syntax-error
     #rx"define-hook: unknown hook on-fit-strat; closest: on-fit-start"
     (let ()
       (define-callback typo () (define-hook (on-fit-strat) (void)))
       typo)))

  (test-case "E2: a hook defined on the wrong side"
    (check-syntax-error
     #rx"define-hook: on-after-batch-transfer is a trainer hook; define-callback cannot define it"
     (let ()
       (define-callback moved ()
         (define-hook (on-after-batch-transfer . batch) (apply values batch)))
       moved))
    (check-syntax-error
     #rx"define-hook: on-exception is a callback hook; define-trainer cannot define it"
     (let ()
       (define-trainer Catching () (define-hook (on-exception e) (void)))
       Catching)))

  (test-case "E3: a hook's arity"
    (check-syntax-error
     #rx": define-hook: on-before-optimizer-step takes 1 argument [(]optimizer[)] at: "
     (let ()
       (define-trainer Stepper () (define-hook (on-before-optimizer-step) (void)))
       Stepper))
    (check-syntax-error
     #rx": define-hook: on-train-batch-end takes 2 arguments [(]loss extra[)] then the batch values at: "
     (let ()
       (define-trainer Ender () (define-hook (on-train-batch-end loss) (void)))
       Ender))
    (check-syntax-error
     #rx": define-hook: on-fit-start takes no arguments at: "
     (let ()
       (define-callback starter () (define-hook (on-fit-start . more) (void)))
       starter))
    (check-syntax-error
     #rx": define-hook: setup takes 1 argument [(]stage[)] at: "
     (let ()
       (define-callback setter () (define-hook (setup stage extra) (void)))
       setter)))

  (test-case "E4: a body form or a hook given twice"
    (check-syntax-error
     #rx"define-trainer: define-optimizers appears twice"
     (let ()
       (define-trainer Twice ()
         (define-optimizers 1)
         (define-optimizers 2))
       Twice))
    (check-syntax-error
     #rx"define-hook: on-fit-start is defined twice"
     (let ()
       (define-callback twice ()
         (define-hook (on-fit-start) 1)
         (define-hook (on-fit-start) 2))
       twice)))

  (test-case "E5: a body form of the other definer"
    (check-syntax-error
     #rx"define-callback: define-training-step is a define-trainer form"
     (let ()
       (define-callback stepping () (define-training-step (x) x))
       stepping))
    (check-syntax-error
     #rx"define-callback: define-optimizers is a define-trainer form"
     (let ()
       (define-callback optimizing () (define-optimizers 1))
       optimizing))
    (check-syntax-error
     #rx"define-callback: define-callbacks is a define-trainer form"
     (let ()
       (define-callback nesting () (define-callbacks 1))
       nesting))
    (check-syntax-error
     #rx"define-callback: define-hparams is a define-trainer form"
     (let ()
       (define-callback tuned (lr) (define-hparams lr))
       tuned))
    (check-syntax-error
     #rx"define-callback: define-forward is a define-trainer form"
     (let ()
       (define-callback forwarding () (define-forward (x) x))
       forwarding))
    (check-syntax-error
     #rx"define-trainer: define-state-key is a define-callback form"
     (let ()
       (define-trainer Keyed () (define-state-key "keyed"))
       Keyed)))

  (test-case "E6: run and log! outside a hook body"
    (check-syntax-error #rx"log!: only allowed inside a trainer or callback hook body"
                        (log! 'loss 1.0))
    (check-syntax-error #rx"log!: only allowed inside a trainer or callback hook body"
                        log!)
    (check-syntax-error #rx"run: only allowed inside a trainer or callback hook body"
                        (run-epoch run)))

  (test-case "E8: define-hparams names fields"
    (check-syntax-error
     #rx"define-hparams: lr is not a field of Classifier"
     (let ()
       (define-trainer Classifier (net momentum)
         (define-hparams momentum lr))
       Classifier)))

  (test-case "body forms are recognised through renamed imports"
    (check-syntax-error
     #rx"define-trainer: define-optimizers appears twice"
     (let ()
       (define-trainer Twice ()
         (opt 1)
         (define-optimizers 2))
       Twice))
    (check-syntax-error
     #rx"define-callback: define-optimizers is a define-trainer form"
     (let ()
       (define-callback optimizing () (opt 1))
       optimizing))
    (check-syntax-error
     #rx"define-hook: unknown hook on-batch-end; closest: on-train-batch-end"
     (let ()
       (define-trainer Typo () (hk (on-batch-end) 1))
       Typo))
    (check-syntax-error
     #rx"define-hook: on-fit-start is defined twice"
     (let ()
       (define-callback twice ()
         (hk (on-fit-start) 1)
         (hk (on-fit-start) 2))
       twice))
    (check-syntax-error
     #rx"define-hparams: lr is not a field of Classifier"
     (let ()
       (define-trainer Classifier (net)
         (hp lr))
       Classifier)))

  (test-case "E9: a body form outside a definer"
    (check-syntax-error
     #rx"define-init: only allowed directly inside define-trainer or define-callback"
     (define-init () (void)))
    (check-syntax-error
     #rx"define-hook: only allowed directly inside define-trainer or define-callback"
     (define-hook (on-fit-start) (void)))
    (check-syntax-error
     #rx"define-training-step: only allowed directly inside define-trainer or define-callback"
     (define-training-step (x) x)))

  (test-case "E10: fields follow define-layer's rules"
    (check-syntax-error
     #rx"define-trainer: with define-init, a field is a bare identifier"
     (let ()
       (define-trainer Defaulted ([lr 0.1]) (define-init () (void)))
       Defaulted))
    (check-syntax-error
     #rx"a field name may not contain a dot"
     (let ()
       (define-callback dotted (ema.decay))
       dotted)))

  (test-case "E11: a callback's kind"
    (check-syntax-error
     #rx"define-callback: expected callback kind [(]plain, monitor or checkpoint[)]"
     (let ()
       (define-callback loud () #:kind loud)
       loud)))

  (test-case "an option of the other definer"
    (check-syntax-error
     #rx"define-callback: #:automatic-optimization[?] is a define-trainer option"
     (let ()
       (define-callback manual () #:automatic-optimization? #f)
       manual))
    (check-syntax-error
     #rx"define-trainer: #:kind is a define-callback option"
     (let ()
       (define-trainer Watching () #:kind monitor)
       Watching)))

  (test-case "log! in a hook body calls run-log! through its contract"
    (define not-a-run 'not-a-run)
    (define (violates-run-log! e)
      (and (exn:fail:contract? e)
           (regexp-match? #rx"^run-log!: contract violation" (exn-message e))
           (regexp-match? #rx"expected: logging-run/c" (exn-message e))))
    (check-exn violates-run-log!
               (lambda ()
                 (syntax-parameterize ([log! (hook-log! #'not-a-run)])
                   (log! 'loss 1.0))))
    (check-exn violates-run-log!
               (lambda ()
                 ((syntax-parameterize ([log! (hook-log! #'not-a-run)]) log!)
                  'loss 1.0 #:on-step? #t))))

  (test-case "a definition that passes the checks constructs"
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
    (check-pred trainer? (Ender 1))
    (check-pred callback? (watcher 1))
    (check-pred classifier? (Classifier 1))))
