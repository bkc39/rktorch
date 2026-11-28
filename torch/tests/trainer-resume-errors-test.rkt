#lang racket/base

(module+ test
  (require (only-in racket/contract exn:fail:contract:blame?)
           (only-in racket/match == match match-define)
           (only-in racket/port open-output-nowhere)
           (only-in racket/string string-contains? string-join)
           rackunit
           (only-in "../main.rkt"
                    current-generator generator-state make-generator tensor->list zeros)
           (only-in "../nn.rkt" Linear adamw mse-loss parameters sgd state-dict step-lr)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "../train/private/resume.rkt" path-noted)
           (only-in "private/checkpoint-kit.rkt" saving-callback Small)
           (only-in "private/comparable.rkt" comparable)
           (only-in "private/trainer-kit.rkt" call-with-logged-warnings toy-loader)
           (only-in "private/wedge.rkt" call-with-process-deadline))

  (define (saved-with dir callbacks #:name [name "e.ckpt"] #:max-epochs [epochs 1]
                      #:weights-only? [weights-only? #f])
    (define path (build-path dir name))
    (void (fit (Small) (toy-loader 4) #:max-epochs epochs #:logger #f
               #:callbacks (append callbacks
                                   (list (saving-callback (lambda (_r) path)
                                                          #:weights-only? weights-only?)))))
    path)

  (define (resume tr path #:max-epochs [epochs 2] #:callbacks [callbacks '()])
    (parameterize ([current-error-port (open-output-nowhere)])
      (fit tr (toy-loader 4) #:max-epochs epochs #:logger #f #:ckpt-path path
           #:callbacks callbacks)))

  (define (check-names-path rx path thunk [message ""] #:kind [kind? exn:fail:contract?])
    (check-exn (lambda (e) (and (kind? e)
                                (regexp-match? rx (exn-message e))
                                (string-contains? (exn-message e) (path->string path))))
               thunk
               message))

  (define-trainer Tuned (net lr)
    (define-init () (set! net (Linear 2 3)) (set! lr 0.9))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "T3-L18: a model that does not load leaves the hyperparameters as they were"
    (with-temporary-directory (dir #:template "hparams-~a")
      (define path (saved-with dir '()))
      (define tr (Tuned))
      (check-names-path #rx"shape|size" path (lambda () (resume tr path)))
      (check-equal? (trainer-hparams tr) (hasheq 'lr 0.9))))

  (define-trainer Adaptive (net lr)
    (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
    (define-hparams lr)
    (define-optimizers (adamw (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Pair (net)
    #:automatic-optimization? #f
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (values (sgd (parameters net) #:lr 0.1) (sgd (parameters net) #:lr 0.1)))
    (define-training-step (x y)
      (for ([opt (in-list (run-optimizers run))])
        (manual-zero-grads! run opt)
        (manual-backward! run (mse-loss (net x) y))
        (manual-step! run opt))
      #f))

  (define-trainer Scheduled (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1))
      (values opt (step-lr opt #:step-size 1)))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define (reshaping key edit)
    (lambda-callback #:state-key (format "reshape-~a" key)
                     #:on-save-checkpoint (lambda (_r tree) (edit tree))))

  (define removed (string->uninterned-symbol "removed"))

  (define (path-set h keys v)
    (match keys
      ['() v]
      [(list k) #:when (eq? v removed) (hash-remove h k)]
      [(cons k more) (hash-set h k (path-set (hash-ref h k #f) more v))]))

  (define ((edit-at keys v) tree)
    (match-define (cons k more) keys)
    (match (path-set (hash-ref tree k #f) more v)
      [(== removed eq?) (hash-remove! tree k)]
      [edited (hash-set! tree k edited)]))

  (define (setup-seen seen)
    (lambda-callback #:setup (lambda (_r _stage) (set-box! seen '(setup)))))

  (define (check-refused-untouched rx path what #:max-epochs [epochs 2])
    (define tr (Small))
    (define weights (comparable (state-dict tr)))
    (define hparams (trainer-hparams tr))
    (define seen (box '()))
    (check-names-path rx path
                      (lambda () (resume tr path #:max-epochs epochs
                                         #:callbacks (list (setup-seen seen))))
                      what)
    (check-equal? (unbox seen) '() (format "~a: no hook ran" what))
    (check-equal? (trainer-hparams tr) hparams (format "~a: the hyperparameters are kept" what))
    (check-equal? (comparable (state-dict tr)) weights (format "~a: the weights are kept" what)))

  (define retuned (reshaping "retuned" (edit-at '("trainer_hparams" "lr") 0.5)))

  (test-case "a resume past #:max-epochs changes nothing"
    (with-temporary-directory (dir #:template "later-~a")
      (define later (saved-with dir (list retuned) #:max-epochs 2))
      (check-refused-untouched #rx"past #:max-epochs 1" later "past #:max-epochs" #:max-epochs 1)))

  (test-case "a resume without the saved current-generator changes nothing"
    (with-temporary-directory (dir #:template "generated-~a")
      (define generated
        (parameterize ([current-generator (make-generator 13)])
          (saved-with dir (list retuned))))
      (check-refused-untouched #rx"current-generator" generated "no current-generator")))

  (test-case "every resume error names the checkpoint's path"
    (with-temporary-directory (dir #:template "named-~a")
      (define path (saved-with dir '()))
      (check-names-path #rx"optimizer.*1.*2" path (lambda () (resume (Pair) path)) "optimizers")
      (check-names-path #rx"does not fit the optimizer" path (lambda () (resume (Adaptive) path))
                        "another optimizer kind")
      (check-names-path #rx"scheduler.*0.*1" path (lambda () (resume (Scheduled) path))
                        "schedulers")
      (define weights (saved-with dir '() #:name "weights.ckpt" #:weights-only? #t))
      (check-names-path #rx"contains only the model" weights (lambda () (resume (Small) weights))
                        "weights-only")
      (define unfit
        (saved-with dir (list (reshaping "unfit" (edit-at '("optimizer_states") (list (hash)))))
                    #:name "unfit.ckpt"))
      (check-names-path #rx"optimizer" unfit (lambda () (resume (Small) unfit))
                        "a contract on a loaded state keeps its blame"
                        #:kind exn:fail:contract:blame?)))

  (define-trainer Tested (net lr)
    (define-init () (set! net (Linear 2 1)) (set! lr 0.1))
    (define-hparams lr)
    (define-test-step (x y) (log! 'test-loss (mse-loss (net x) y))))

  (define (epoch-entry . fields)
    (define well-formed (hash "logger" #t "both" #f "value" 1.0 "weight" 2 "reduce" 'mean))
    (list '("metric_book" "train" "epoch") (hash "k" (apply hash-set* well-formed fields))))

  (define malformed-entries
    (list (list '("loops") removed)
          (list '("loops") (hash))
          (list '("loops" "fit_loop" "phase") 3)
          (list '("loops" "fit_loop" "eval_batches") "many")
          (list '("loops" "fit_loop" "epoch_end_logged") 1)
          (list '("trainer_hparams") removed)
          (list '("trainer_hparams") '(lr))
          (list '("global_step") removed)
          (list '("callback_metrics") (hash "x" "y"))
          (list '("rng") removed)
          (list '("rng" "racket") '(1 2))
          (list '("rng" "racket") (list 1 2 3 4 5 (expt 2 40)))
          (list '("rng" "cpu") (zeros 3))
          (list '("rng" "cpu") (zeros 5056 #:dtype 'uint8))
          (list '("rng" "cuda") (zeros 3))
          (list '("rng" "current_generator") 5)
          (list '("metric_book") (hash "train" 1))
          (epoch-entry "value" "x")
          (epoch-entry "value" (zeros 2))
          (epoch-entry "reduce" "mean")
          (epoch-entry "weight" -1)
          (epoch-entry "logger" 1)
          (list '("metric_book" "eval" "steps")
                (hash "k" (hash "logger" #t "both" "no" "value" 1.0)))
          (list '("metric_book" "train" "stepped") (hash "k" 1))
          (list '("metric_book" "train" "closed") (hash "k" "x"))
          (list '("MixedPrecision") (hash "scale" -1.0))
          (list '("optimizer_states") (hash))
          (list '("optimizer_states") '(1))
          (list '("lr_schedulers") 0)
          (list '("lr_schedulers") '(1))))

  (with-temporary-directory (dir #:template "malformed-~a")
    (for ([malformed (in-list malformed-entries)] [i (in-naturals)])
      (match-define (list keys v) malformed)
      (define key (string-join keys "."))
      (test-case (format "T3-L3: ~a as ~e is refused before any hook, naming the path and the key"
                         key v)
        (call-with-process-deadline
         60
         #:what (format "refusing a malformed ~a" key)
         (lambda ()
           (define path
             (saved-with dir (list (reshaping i (edit-at keys v))) #:name (format "~a.ckpt" i)))
           (define seen (box '()))
           (check-names-path (regexp (string-append "cannot be resumed.*key: \""
                                                    (regexp-quote key)))
                             path
                             (lambda ()
                               (resume (Small) path
                                       #:callbacks (list (lambda-callback
                                                          #:setup (lambda (_r _stage)
                                                                    (set-box! seen '(setup)))))))
                             key)
           (check-equal? (unbox seen) '() "no hook ran")
           (check-pred exact-nonnegative-integer? (random 10)
                       "the Racket PRNG still answers after the refusals")))))
    (test-case "T3-L3: a metric_book is checked in a file without optimizer states"
      (define path
        (saved-with dir (list (reshaping "book-only" (edit-at '("metric_book") (hash "train" 1)))
                              (reshaping "no-optimizers" (edit-at '("optimizer_states") removed)))
                    #:name "book-only.ckpt"))
      (check-names-path #rx"cannot be resumed.*key: \"metric_book" path
                        (lambda () (resume (Small) path))))
    (test-case "T3-L3: test #:ckpt-path refuses a malformed tree"
      (define path
        (saved-with dir (list (reshaping "test" (edit-at '("trainer_hparams") removed)))
                    #:name "test.ckpt"))
      (check-names-path #rx"cannot be resumed.*trainer_hparams" path
                        (lambda () (test (Tested) (toy-loader 2) #:logger #f #:ckpt-path path)))))

  (test-case "a weight-0 book entry, as a zero-row batch writes, is resumed"
    (with-temporary-directory (dir #:template "weightless-~a")
      (for ([reduce (in-list '(sum mean))])
        (match-define (list keys v) (epoch-entry "weight" 0 "reduce" reduce))
        (define path
          (saved-with dir (list (reshaping "weightless" (edit-at keys v)))
                      #:name (format "~a.ckpt" reduce)))
        (check-pred run? (resume (Small) path) (format "reduce ~a" reduce)))))

  (test-case "a saved current-generator state with no current-generator to go to warns"
    (with-temporary-directory (dir #:template "cleared-~a")
      (define path
        (parameterize ([current-generator (make-generator 13)])
          (saved-with dir '())))
      (define clearing (lambda-callback #:on-fit-start (lambda (_r) (current-generator #f))))
      (define-values (_run warnings)
        (parameterize ([current-generator (make-generator 13)])
          (call-with-logged-warnings
           (lambda () (resume (Small) path #:callbacks (list clearing))))))
      (check-match warnings (list (regexp #rx"current-generator.*not restored")))))

  (test-case "a malformed current-generator state is refused before any hook, the generator kept"
    (call-with-process-deadline
     60
     #:what "refusing a malformed current-generator state"
     (lambda ()
       (with-temporary-directory (dir #:template "generator-state-~a")
         (define g (make-generator 13))
         (define zeroed
           (edit-at '("rng" "current_generator" "state") (zeros 5056 #:dtype 'uint8)))
         (define path
           (parameterize ([current-generator g])
             (saved-with dir (list (reshaping "zeroed" zeroed)))))
         (define tr (Small))
         (define before (tensor->list (generator-state g)))
         (define seen (box '()))
         (parameterize ([current-generator g])
           (check-names-path #rx"cannot be resumed.*key: \"rng.current_generator.state" path
                             (lambda () (resume tr path #:callbacks (list (setup-seen seen))))))
         (check-equal? (unbox seen) '() "no hook ran")
         (check-equal? (tensor->list (generator-state g)) before "the generator is kept")))))

  (struct exn:fail:opaque exn:fail ())

  (test-case "an exception whose struct this module cannot rebuild comes back unchanged, silently"
    (define e (exn:fail:opaque "raised" (current-continuation-marks)))
    (define-values (noted warnings)
      (call-with-logged-warnings (lambda () (path-noted e (string->path "e.ckpt")))))
    (check-eq? noted e)
    (check-equal? warnings '()))

  (test-case "an epoch-end file without epoch_end_logged resumes without a warning"
    (with-temporary-directory (dir #:template "older-~a")
      (define drop-logged (edit-at '("loops" "fit_loop" "epoch_end_logged") removed))
      (define path (saved-with dir (list (reshaping "older" drop-logged))))
      (define-values (_run warnings) (call-with-logged-warnings (lambda () (resume (Small) path))))
      (check-equal? warnings '()))))
