#lang racket/base

(require (only-in racket/list filter-not)
         (only-in racket/match match match* match-define)
         (only-in racket/string string-join)
         (only-in "../../nn/grad-scaler.rkt" load-grad-scaler-state!)
         (only-in "../../nn/optim-state.rkt" load-optimizer-state!)
         (only-in "../../nn/scheduler.rkt" load-scheduler-state!)
         (only-in "../../nn/state-dict.rkt" load-state!)
         (only-in "../optimization.rkt" set-hparam! trainer-hparams)
         (only-in "book.rkt" restore-book-state!)
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

(define (fit-loop tree)
  (hash-ref (hash-ref tree "loops") "fit_loop"))

(define (checked who path tree keys ok?)
  (define v
    (for/fold ([v tree]) ([k (in-list keys)])
      (and (hash? v) (hash-ref v k #f))))
  (unless (ok? v)
    (raise-arguments-error
     who "the checkpoint's entry is missing or malformed, so it cannot be resumed"
     "key" (string-join keys ".")
     "path" path))
  v)

(define (string-keyed? h)
  (and (hash? h) (andmap string? (hash-keys h))))

(define (check-resumable! path tree)
  (checked 'fit path tree '("global_step") exact-nonnegative-integer?)
  (checked 'fit path tree '("callback_metrics") string-keyed?)
  (for ([key (in-list '("epoch" "batches_that_stepped" "eval_batches"))])
    (checked 'fit path tree (list "loops" "fit_loop" key) exact-nonnegative-integer?))
  (match (checked 'fit path tree '("loops" "fit_loop" "phase") string?)
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
  (checked who path tree '("trainer_hparams") string-keyed?)
  (when (eq? who 'fit)
    (check-resumable! path tree))
  (source path tree))

(define (model-entry name)
  (and (regexp-match? #rx"^state_dict[.]" name)
       (substring name (string-length "state_dict."))))

(define (restore-callbacks! r tree)
  (match (hash-ref tree "callbacks" #f)
    [#f (void)]
    [states
     (define cbs (run-callbacks r))
     (define keys (map callback-state-key cbs))
     (match (sort (filter-not (lambda (k) (member k keys)) (hash-keys states)) string<?)
       ['() (void)]
       [orphans
        (warn-train! (format (string-append "~a: the checkpoint holds state for callbacks this run"
                                            " does not have, which is ignored: ~s")
                             (run-entry r) orphans))])
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

(define (restore-modules! r ckpt)
  (match-define (source path tree) ckpt)
  (define tr (run-trainer r))
  (define before (trainer-hparams tr))
  (with-handlers ([(lambda (_) #t)
                   (lambda (e)
                     (for ([(field v) (in-hash before)])
                       (set-hparam! tr field v))
                     (raise e))])
    (restore-hparams! r tree)
    (call-trainer r 'on-load-checkpoint tree)
    (load-state! tr path #:rename model-entry))
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
    (load! target state)))

(define (restore-training-state! r ckpt max-epochs)
  (match-define (source path tree) ckpt)
  (match* ((run-grad-scaler r) (hash-ref tree "MixedPrecision" #f))
    [((? values scaler) (? values state)) (load-grad-scaler-state! scaler state)]
    [(_ _) (void)])
  (match-define (hash-table ("phase" phase) ("epoch" saved-epoch)
                            ("batches_that_stepped" stepped) ("eval_batches" evals))
    (fit-loop tree))
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
