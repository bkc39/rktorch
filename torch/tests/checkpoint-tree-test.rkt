#lang racket/base

(module+ test
  (require (only-in racket/contract/base flat-contract-predicate)
           (only-in racket/function curry curryr)
           rackunit
           (only-in racket/match match-define)
           (only-in "../generated.rkt" sum-dim-intlist)
           (only-in "../main.rkt"
                    current-generator make-generator ones rng-state tensor transpose)
           (only-in "../nn.rkt"
                    Linear load-optimizer-state! mse-loss optimizer-state-dict parameters
                    scheduler-state-dict sgd step-lr)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" call-with-live-run check-exn/contract Small stored)
           (only-in "private/child.rkt" run-child)
           (only-in "private/comparable.rkt" comparable host-bytes)
           (only-in "private/safetensors-reader.rkt" read-safetensors)
           (only-in "private/trainer-kit.rkt" toy-loader))

  (define-trainer Scheduled (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
      (values opt (step-lr opt #:step-size 1 #:gamma 0.5)))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y)))

  (define-trainer Spread (net)
    (define-init (width #:bias [b? #t] . more) (set! net (Linear 2 width #:bias? b?)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y)))

  (define (saved-at path #:trainer [tr (Small)] #:callbacks [callbacks '()]
                    #:weights-only? [weights-only? #f])
    (define seen (box #f))
    (void (call-with-live-run
           #:trainer tr #:callbacks callbacks
           (lambda (r)
             (save-checkpoint! r path #:weights-only? weights-only?)
             (set-box! seen r))))
    (values (read-checkpoint path) (unbox seen)))

  (test-case "LO-9: the training state is saved as the nn accessors give it"
    (with-temporary-directory (dir #:template "tree-~a")
      (define tr (Scheduled))
      (define-values (tree r) (saved-at (build-path dir "s.ckpt") #:trainer tr))
      (define opt (car (run-optimizers r)))
      (define sched (car (run-schedulers r)))
      (check-equal? (comparable (hash-ref tree "optimizer_states"))
                    (comparable (list (optimizer-state-dict opt tr))))
      (check-equal? (hash-ref tree "lr_schedulers")
                    (list (scheduler-state-dict (scheduler-config-scheduler sched))))
      (check-false (hash-has-key? tree "MixedPrecision") "no grad scaler, no entry")
      (check-equal? (hash-ref tree "trainer_hparams") (hash))
      (define loop (hash-ref (hash-ref tree "loops") "fit_loop"))
      (check-equal? (sort (hash-keys loop) string<?)
                    '("batches_that_stepped" "epoch" "eval_batches" "phase"))
      (check-equal? (map (curry hash-ref loop) '("epoch" "phase" "batches_that_stepped"))
                    '(1 "fit-end" 2))
      (define fresh (sgd (parameters tr) #:lr 0.5))
      (load-optimizer-state! fresh tr (car (hash-ref tree "optimizer_states")))
      (check-equal? (comparable (optimizer-state-dict fresh tr))
                    (comparable (optimizer-state-dict opt tr))
                    "the saved optimizer state loads back")))

  (test-case "LO-9: a mid-epoch save records its phase and batch"
    (with-temporary-directory (dir #:template "phase-~a")
      (define path (build-path dir "b.ckpt"))
      (define saver
        (lambda-callback
         #:on-train-batch-end
         (lambda (r . _)
           (when (= (run-global-step r) 1) (save-checkpoint! r path)))))
      (void (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f #:callbacks (list saver)))
      (define loop (hash-ref (hash-ref (read-checkpoint path) "loops") "fit_loop"))
      (check-equal? (hash-ref loop "phase") "batch-end")
      (check-equal? (hash-ref loop "batch_index") 0)
      (check-equal? (hash-ref loop "epoch") 0)))

  (test-case "LO-9: a validation-end save keeps the training loop's batch index"
    (with-temporary-directory (dir #:template "val-end-~a")
      (define saves (box '()))
      (define saver
        (lambda-callback
         #:on-validation-end
         (lambda (r)
           (define path (build-path dir (format "v~a.ckpt" (length (unbox saves)))))
           (save-checkpoint! r path)
           (set-box! saves (cons path (unbox saves))))))
      (void (fit (Small) (toy-loader 8) #:val (toy-loader 2) #:max-epochs 1 #:logger #f
                 #:val-check-interval 2 #:num-sanity-val-steps 0 #:callbacks (list saver)))
      (define loops
        (for/list ([path (in-list (reverse (unbox saves)))])
          (hash-ref (hash-ref (read-checkpoint path) "loops") "fit_loop")))
      (check-equal? (map (curryr hash-ref "phase") loops) '("validation-end" "validation-end"))
      (check-equal? (map (curryr hash-ref "batch_index" #f) loops) '(1 3))))

  (test-case "weights-only keeps the model, hyperparameters and position, as Lightning's"
    (with-temporary-directory (dir #:template "weights-~a")
      (define-values (tree _r) (saved-at (build-path dir "w.ckpt") #:weights-only? #t))
      (check-equal? (sort (hash-keys tree) string<?)
                    '("callback_metrics" "epoch" "global_step" "hyper_parameters"
                      "init_dropped" "init_formals" "loops" "state_dict" "trainer_hparams"))))

  (test-case "the RNG bundle holds the CPU stream, current-generator and Racket's PRNG"
    (with-temporary-directory (dir #:template "rng-~a")
      (define path (build-path dir "r.ckpt"))
      (define expected (box #f))
      (define (capture-then-save r)
        (set-box! expected (list (host-bytes (rng-state))
                                 (vector->list (pseudo-random-generator->vector
                                                (current-pseudo-random-generator)))))
        (save-checkpoint! r path))
      (void (call-with-live-run capture-then-save))
      (define rng (hash-ref (read-checkpoint path) "rng"))
      (check-equal? (list (host-bytes (hash-ref rng "cpu")) (hash-ref rng "racket"))
                    (unbox expected))
      (check-false (hash-ref rng "current_generator"))
      (parameterize ([current-generator (make-generator 7)])
        (void (call-with-live-run (lambda (r) (save-checkpoint! r path)))))
      (check-equal? (hash-ref (hash-ref (hash-ref (read-checkpoint path) "rng") "current_generator")
                              "device")
                    'cpu)))

  (test-case "B6: the init formals keep their shape; the file goes where the path says"
    (with-temporary-directory (dir #:template "formals-~a")
      (define path (build-path dir "deeper" "still" "f.ckpt"))
      (define-values (tree _r) (saved-at path #:trainer (Spread 3 #:bias #f 'x "y")))
      (check-equal? (hash-ref tree "init_formals")
                    '(("positional" "width") ("keyword" "#:bias" "b?") ("rest" "more")))
      (check-equal? (hash-ref tree "hyper_parameters") (hash "width" 3 "b?" #f "more" '(x "y")))
      (check-equal? (hash-ref tree "init_dropped") '())
      (define seen (box #f))
      (void (load-from-checkpoint (lambda (w #:bias b . more) (set-box! seen (list* w b more))
                                    (Spread w #:bias b))
                                  path
                                  #:init-args (hasheq 'width 3 'unknown 1)))
      (check-equal? (unbox seen) '(3 #f)
                    "an #:init-args name that is no formal is ignored, and saved rest values too")))

  (test-case "a callback state-dict that is not a checkpoint value names the callback"
    (with-temporary-directory (dir #:template "bad-state-~a")
      (define bad (lambda-callback #:state-key "holder" #:state-dict (lambda (_r) (vector 1))))
      (check-exn/contract #rx"state-dict.*\"holder\""
                          (lambda ()
                            (call-with-live-run
                             #:callbacks (list bad)
                             (lambda (r) (save-checkpoint! r (build-path dir "x.ckpt"))))))
      (check-equal? (directory-list dir) '())))

  (define (save-raises dir rx #:callbacks [callbacks '()])
    (check-exn/contract rx
                        (lambda ()
                          (call-with-live-run
                           #:callbacks callbacks
                           (lambda (r) (save-checkpoint! r (build-path dir "x.ckpt"))))))
    (check-equal? (directory-list dir) '() "nothing is written"))

  (test-case "a top-level key a hook adds must be a string"
    (with-temporary-directory (dir #:template "top-key-~a")
      (define note
        (lambda-callback #:on-save-checkpoint (lambda (_r ckpt) (hash-set! ckpt 'note 1))))
      (save-raises dir #rx"top-level key.*key path: '\\(note\\)" #:callbacks (list note))))

  (define-trainer Indexed (net index)
    (define-init (ix) (set! index ix) (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y)))

  (test-case "a tensor of a dtype the format has no tag for is no checkpoint value, and no raise"
    (define narrow (sum-dim-intlist (tensor '(1.0)) #f #f 'int32))
    (check-false ((flat-contract-predicate checkpoint-value/c) (hash "x" narrow)))
    (with-temporary-directory (dir #:template "narrow-~a")
      (save-raises dir #rx"key path: '\\(\"probe\" \"x\"\\)"
                   #:callbacks (list (stored (box (hash "x" narrow)))))
      (define-values (tree _r) (saved-at (build-path dir "k.ckpt") #:trainer (Indexed narrow)))
      (check-equal? (hash-ref tree "init_dropped") '("ix"))))

  (test-case "a native fault in the value check reaches the caller, not a #f"
    (define-values (status answer stderr)
      (run-child '(begin
                    (require (only-in ffi/unsafe _intptr cast)
                             (only-in racket/contract/base flat-contract-predicate)
                             (only-in torch/foreign/raw/syntax _Tensor)
                             (only-in torch/foreign/structs tensor-impl)
                             (only-in torch/train checkpoint-value/c))
                    (define bad (tensor-impl (cast 8 _intptr _Tensor) '(2 2)))
                    (write (with-handlers ([exn:fail? (lambda (e) (list 'raised (exn-message e)))])
                             (list 'returned ((flat-contract-predicate checkpoint-value/c) bad)))))
                 #:environment (list (cons "RKTORCH_ON_NATIVE_FAULT" #f))
                 #:within 60))
    (check-equal? status 0 stderr)
    (check-match answer (list 'raised (regexp #rx"invalid memory reference"))))

  (test-case "a value shared down 40 levels is refused before it is expanded"
    (with-temporary-directory (dir #:template "shared-deep-~a")
      (define (doubled depth) (for/fold ([v '(1)]) ([_ (in-range depth)]) (list v v)))
      (save-raises dir #rx"100 MB" #:callbacks (list (stored (box (doubled 40)))))
      (define-values (tree _r) (saved-at (build-path dir "d.ckpt")
                                         #:callbacks (list (stored (box (doubled 8))))))
      (check-equal? (hash-ref tree "probe") (doubled 8))))

  (test-case "a tensor held twice is written once and reads back shared"
    (with-temporary-directory (dir #:template "shared-~a")
      (define t (ones 2 3))
      (define view (transpose t 0 1))
      (define (unnamed-entries path)
        (for/list ([k (in-hash-keys (read-safetensors path))] #:when (regexp-match? #rx"^[$]" k))
          k))
      (define path (build-path dir "t.ckpt"))
      (define once (build-path dir "once.ckpt"))
      (define-values (tree _r) (saved-at path #:callbacks (list (stored (box (list t t view))))))
      (define-values (_once _run) (saved-at once #:callbacks (list (stored (box (list t view))))))
      (check-equal? (length (unnamed-entries path)) (length (unnamed-entries once)))
      (match-define (list a b c) (hash-ref tree "probe"))
      (check-eq? a b)
      (check-not-eq? a c)
      (check-equal? (host-bytes c) (host-bytes view))))

  (define-trainer Loading (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (x y) (mse-loss (net x) y))
    (define-hook (on-load-checkpoint _ckpt)
      (set-box! load-refusal (with-handlers ([exn:fail:contract? exn-message])
                               (save-checkpoint! run "never.ckpt")))))

  (define load-refusal (box #f))

  (test-case "an exception-phase save names the training batch that failed"
    (with-temporary-directory (dir #:template "exception-~a")
      (define path (build-path dir "x.ckpt"))
      (define seen (box 0))
      (check-exn #rx"boom"
                 (lambda ()
                   (fit (Small) (toy-loader 6) #:max-epochs 1 #:logger #f
                        #:callbacks
                        (list (lambda-callback
                               #:on-train-batch-end
                               (lambda (_r . _)
                                 (set-box! seen (add1 (unbox seen)))
                                 (when (= (unbox seen) 2) (error "boom")))
                               #:on-exception (lambda (r _e) (save-checkpoint! r path)))))))
      (define loop (hash-ref (hash-ref (read-checkpoint path) "loops") "fit_loop"))
      (check-equal? (map (curry hash-ref loop) '("phase" "batch_index")) '("exception" 1))))

  (test-case "a run that has not started is not saved, as Lightning's"
    (with-temporary-directory (dir #:template "unstarted-~a")
      (define path (build-path dir "s.ckpt"))
      (define-values (_tree _run) (saved-at path #:trainer (Loading)))
      (void (load-from-checkpoint Loading path))
      (check-regexp-match #rx"save-checkpoint!.*contract violation" (unbox load-refusal))
      (check-equal? (directory-list dir) (list (string->path "s.ckpt")))))

  (test-case "a run saved after fit has returned is the one saved at its end, as Lightning's"
    (with-temporary-directory (dir #:template "after-~a")
      (define at-end (build-path dir "end.ckpt"))
      (define after (build-path dir "after.ckpt"))
      (define r
        (fit (Small) (toy-loader 4) #:max-epochs 2 #:logger #f
             #:callbacks (list (lambda-callback
                                #:on-fit-end (lambda (r) (save-checkpoint! r at-end))))))
      (save-checkpoint! r after)
      (define-values (end back) (values (read-checkpoint at-end) (read-checkpoint after)))
      (check-equal? (hash-ref (hash-ref (hash-ref back "loops") "fit_loop") "phase") "fit-end")
      (for ([key (in-list '("loops" "epoch" "global_step" "hyper_parameters" "lr_schedulers"))])
        (check-equal? (hash-ref back key) (hash-ref end key) key))
      (check-equal? (comparable (hash-ref back "optimizer_states"))
                    (comparable (hash-ref end "optimizer_states")))
      (define failed (box #f))
      (check-exn #rx"boom"
                 (lambda ()
                   (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f
                        #:callbacks (list (lambda-callback
                                           #:on-train-batch-end
                                           (lambda (r . _) (set-box! failed r) (error "boom")))))))
      (save-checkpoint! (unbox failed) after)
      (check-equal? (hash-ref (hash-ref (hash-ref (read-checkpoint after) "loops") "fit_loop")
                              "phase")
                    "other"
                    "a failed fit's run is saved, but not as resumable"))))
