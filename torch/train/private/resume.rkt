#lang racket/base

(require (only-in racket/match match match-define)
         (only-in racket/string string-join)
         (only-in "../../nn/grad-scaler.rkt" load-grad-scaler-state!)
         (only-in "../../nn/optim-state.rkt" load-optimizer-state!)
         (only-in "../../nn/scheduler.rkt" load-scheduler-state!)
         (only-in "../../nn/state-dict.rkt" load-state!)
         (only-in "../optimization.rkt" set-hparam! trainer-hparams)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "book.rkt" restore-book-state! saved-book?)
         (only-in "dispatch.rkt" call-callback call-callbacks call-trainer warn-train!)
         (only-in "format.rkt" read-checkpoint-file)
         (only-in "restore.rkt" restore-rng-bundle!)
         (only-in "state.rkt"
                  run-book run-callbacks run-device run-entry run-grad-scaler run-optimizers
                  run-scheduler-configs run-trainer set-run-batches-stepped! set-run-epoch!
                  set-run-eval-batches! set-run-global-step!)
         (only-in "types.rkt" callback-state-key not-implemented scheduler-config-scheduler))

(provide checkpoint-source
         restore-modules!
         restore-rng!
         restore-training-state!)

(struct source (path tree))

(define fit-loop-path '("loops" "fit_loop"))

(define (tree-ref tree keys)
  (for/fold ([v tree]) ([k (in-list keys)])
    (and (hash? v) (hash-ref v k #f))))

(define (fit-loop tree) (tree-ref tree fit-loop-path))

(define (checked-ref who path tree keys ok?)
  (define v (tree-ref tree keys))
  (unless (ok? v)
    (raise-arguments-error
     who "the checkpoint's entry is missing or malformed, so it cannot be resumed"
     "key" (string-join keys ".")
     "path" path))
  v)

(define (string-keyed? h)
  (and (hash? h) (andmap string? (hash-keys h))))

(define (metrics-table? h)
  (and (string-keyed? h) (andmap real? (hash-values h))))

(define (check-full! path tree)
  (when (hash-has-key? tree "optimizer_states")
    (for ([keys (in-list '(("optimizer_states") ("lr_schedulers") ("rng" "racket")))])
      (checked-ref 'fit path tree keys list?))
    (checked-ref 'fit path tree '("rng" "cpu") tensor?)
    (when (hash-has-key? tree "metric_book")
      (checked-ref 'fit path tree '("metric_book") saved-book?))))

(define (check-resumable! path tree)
  (define (check! keys ok?) (checked-ref 'fit path tree keys ok?))
  (check! '("global_step") exact-nonnegative-integer?)
  (check! '("callback_metrics") metrics-table?)
  (for ([key (in-list '("epoch" "batches_that_stepped" "eval_batches"))])
    (check! (append fit-loop-path (list key)) exact-nonnegative-integer?))
  (check! (append fit-loop-path '("epoch_end_logged")) boolean?)
  (check-full! path tree)
  (match (check! (append fit-loop-path '("phase")) string?)
    [(or "epoch-end" "fit-end") (void)]
    [(and phase (or "batch-end" "validation-end" "exception"))
     (not-implemented (format "fit: resuming from a mid-epoch (~a) checkpoint" phase) 'T3)]
    [phase
     (raise-arguments-error
      'fit (string-append "the checkpoint was saved outside an epoch end, a batch end, a validation"
                          " end, an exception and the fit's end, so it cannot be resumed")
      "phase" phase
      "path" path)]))

(define (checkpoint-source who path)
  (define tree (read-checkpoint-file who path))
  (checked-ref who path tree '("trainer_hparams") string-keyed?)
  (when (eq? who 'fit)
    (check-resumable! path tree))
  (source path tree))

(define (model-entry name)
  (and (regexp-match? #rx"^state_dict[.]" name)
       (substring name (string-length "state_dict."))))

(define (warn-orphan-callbacks! r states)
  (define orphans
    (sort (remove* (map callback-state-key (run-callbacks r)) (hash-keys states)) string<?))
  (unless (null? orphans)
    (warn-train! (format (string-append "~a: the checkpoint holds state for callbacks this run"
                                        " does not have, which is ignored: ~s")
                         (run-entry r) orphans))))

(define (restore-callbacks! r tree)
  (match (hash-ref tree "callbacks" #f)
    [#f (void)]
    [states
     (define cbs (run-callbacks r))
     (warn-orphan-callbacks! r states)
     (call-callbacks r 'on-load-checkpoint cbs tree)
     (for ([cb (in-list cbs)])
       (define key (callback-state-key cb))
       (when (hash-has-key? states key)
         (call-callback r cb 'load-state! (list (hash-ref states key)))))]))

(define (restore-hparams! r tree)
  (define tr (run-trainer r))
  (define fields (trainer-hparams tr))
  (for ([(name v) (in-hash (hash-ref tree "trainer_hparams"))])
    (define field (string->symbol name))
    (if (hash-has-key? fields field)
        (set-hparam! tr field v)
        (warn-train! (format (string-append "~a: the checkpoint's hyperparameter ~a is not a"
                                            " define-hparams field of this trainer, so it is"
                                            " ignored")
                             (run-entry r) field)))))

(define (call-naming-path path thunk)
  (with-handlers ([exn:fail?
                   (lambda (e)
                     (define message (format "~a\n  path: ~e" (exn-message e) path))
                     (raise ((if (exn:fail:contract? e) exn:fail:contract exn:fail)
                             message (exn-continuation-marks e))))])
    (thunk)))

(define (call-putting-hparams-back-on-escape tr thunk)
  (define before (trainer-hparams tr))
  (with-handlers ([(lambda (_) #t)
                   (lambda (e)
                     (for ([(field v) (in-hash before)])
                       (set-hparam! tr field v))
                     (raise e))])
    (thunk)))

(define (restore-modules! r ckpt)
  (match-define (source path tree) ckpt)
  (define tr (run-trainer r))
  (call-putting-hparams-back-on-escape
   tr
   (lambda ()
     (restore-hparams! r tree)
     (call-trainer r 'on-load-checkpoint tree)
     (call-naming-path path (lambda () (load-state! tr path #:rename model-entry)))))
  (restore-callbacks! r tree))

(define (restore-each! path tree key what targets load!)
  (define states
    (hash-ref tree key
              (lambda ()
                (raise-arguments-error
                 'fit (format (string-append "trying to restore ~a state, but the checkpoint"
                                             " contains only the model; it was probably saved"
                                             " with #:weights-only? #t")
                              what)
                 "path" path))))
  (unless (= (length states) (length targets))
    (raise-arguments-error
     'fit (format "~a states: the checkpoint holds ~a and the trainer configures ~a"
                  what (length states) (length targets))
     "path" path))
  (for ([target (in-list targets)] [state (in-list states)])
    (call-naming-path path (lambda () (load! target state)))))

(define (restore-training-state! r ckpt max-epochs)
  (match-define (source path tree) ckpt)
  (define scaler (run-grad-scaler r))
  (define scaler-state (hash-ref tree "MixedPrecision" #f))
  (when (and scaler scaler-state)
    (load-grad-scaler-state! scaler scaler-state))
  (match-define (hash-table ("phase" phase) ("epoch" saved-epoch)
                            ("batches_that_stepped" stepped) ("eval_batches" evals))
    (fit-loop tree))
  (unless (hash-ref (fit-loop tree) "epoch_end_logged" #t)
    (warn-train! (format (string-append "fit: the checkpoint ~a was saved before the trainer's"
                                        " on-train-epoch-end ran, so the values logged there"
                                        " are not in it and the resume is not exact; save from"
                                        " a #:kind checkpoint callback or model-checkpoint for"
                                        " an exact resume")
                         path)))
  (define epoch (if (equal? phase "epoch-end") (add1 saved-epoch) saved-epoch))
  (set-run-epoch! r epoch)
  (set-run-global-step! r (hash-ref tree "global_step"))
  (set-run-batches-stepped! r stepped)
  (set-run-eval-batches! r evals)
  (restore-book-state! (run-book r)
                       (hash-ref tree "callback_metrics")
                       (hash-ref tree "metric_book" #f)
                       #:close-train-epoch? (equal? phase "epoch-end"))
  (when (and max-epochs (> epoch max-epochs))
    (raise-arguments-error
     'fit (format "the checkpoint resumes at epoch ~a, past #:max-epochs ~a" epoch max-epochs)
     "path" path))
  (define tr (run-trainer r))
  (restore-each! path tree "optimizer_states" "optimizer" (run-optimizers r)
                 (lambda (opt state) (load-optimizer-state! opt tr state)))
  (restore-each! path tree "lr_schedulers" "scheduler"
                 (map scheduler-config-scheduler (run-scheduler-configs r))
                 load-scheduler-state!))

(define (restore-rng! r ckpt)
  (match-define (source path tree) ckpt)
  (restore-rng-bundle! (hash-ref tree "rng") (run-device r) #:path path))
