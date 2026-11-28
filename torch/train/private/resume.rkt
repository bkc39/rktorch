#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre ...+ expr))
         (only-in racket/contract contract-first-order-passes?)
         (only-in racket/match match match-define)
         (only-in racket/string string-join)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in threading ~>)
         (only-in "../../foreign.rkt" cpu-device dtype shape tensor?)
         (only-in "../../foreign/generator.rkt"
                  current-generator generator-device make-generator set-generator-state!)
         (only-in "../../nn/grad-scaler.rkt" grad-scaler-state/c load-grad-scaler-state!)
         (only-in "../../nn/optim-state.rkt" load-optimizer-state!)
         (only-in "../../nn/scheduler.rkt" load-scheduler-state!)
         (only-in "../../nn/state-dict.rkt" load-state!)
         (only-in "../optimization.rkt" set-hparam! trainer-hparams)
         (only-in "book.rkt" restore-book-state! saved-book-checks)
         (only-in "dispatch.rkt" call-callback call-callbacks call-trainer warn-train!)
         (only-in "format.rkt" read-checkpoint-file)
         (only-in "restore.rkt" check-current-generator! restore-rng-bundle!)
         (only-in "state.rkt"
                  run-book run-callbacks run-device run-entry run-grad-scaler run-optimizers
                  run-scheduler-configs run-trainer set-run-batches-stepped! set-run-epoch!
                  set-run-eval-batches! set-run-global-step!)
         (only-in "types.rkt" callback-state-key not-implemented scheduler-config-scheduler))

(provide check-resume-epoch!
         checkpoint-source
         path-noted
         restore-modules!
         restore-rng!
         restore-training-state!)

(struct source (path tree))

(define fit-loop-path '("loops" "fit_loop"))

(define (tree-ref tree keys)
  (for/fold ([v tree]) ([k (in-list keys)])
    (and (hash? v) (hash-ref v k #f))))

(define (fit-loop tree) (tree-ref tree fit-loop-path))

(define (fit-loop-key key) (append fit-loop-path (list key)))

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

(define (states? v)
  (and (list? v) (andmap hash? v)))

(define (racket-generator-state? v)
  (and (list? v)
       (with-handlers ([exn:fail? (lambda (_) #f)])
         (~> (list->vector v) vector->pseudo-random-generator pseudo-random-generator?))))

(define ((generator-accepts? device) v)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (set-generator-state! (make-generator 0 #:device device) v)
    #t))

(define ((device-stream-state? type) v)
  (and (tensor? v)
       (eq? (dtype v) 'uint8)
       (if (equal? type "cuda")
           (equal? (shape v) '(16))
           (= 1 (length (shape v))))))

(define rng-streams '("cpu" "current_generator" "racket"))

(define (saved-generator? v)
  (match v
    [#f #t]
    [(hash-table ("device" _) ("state" (? tensor?))) #t]
    [_ #f]))

(define (scaler-state? v)
  (contract-first-order-passes? grad-scaler-state/c v))

(define (check-rng! check! bundle)
  (define (check-stream! name ok?) (check! (list "rng" name) ok?))
  (check-stream! "racket" racket-generator-state?)
  (check-stream! "cpu" (generator-accepts? (cpu-device)))
  (for ([type (in-list (remove* rng-streams (hash-keys bundle)))])
    (check-stream! type (device-stream-state? type)))
  (check-stream! "current_generator" saved-generator?))

(define (check-live-generator! check! path saved)
  (check-current-generator! saved path)
  (when saved
    (check! '("rng" "current_generator" "state")
            (~> (current-generator) generator-device generator-accepts?))))

(define (check-book! check!)
  (for ([book-check (in-list saved-book-checks)])
    (match-define (cons keys ok?) book-check)
    (check! (cons "metric_book" keys) ok?)))

(define (check-training-state! check! path tree)
  (when (hash-has-key? tree "optimizer_states")
    (check! '("optimizer_states") states?)
    (check! '("lr_schedulers") states?)
    (check-rng! check! (check! '("rng") hash?))
    (check-live-generator! check! path (tree-ref tree '("rng" "current_generator"))))
  (when (hash-has-key? tree "metric_book")
    (check-book! check!))
  (when (hash-has-key? tree "MixedPrecision")
    (check! '("MixedPrecision") scaler-state?)))

(define (check-resumable! path tree)
  (define (check! keys ok?) (checked-ref 'fit path tree keys ok?))
  (check! '("global_step") exact-nonnegative-integer?)
  (check! '("callback_metrics") metrics-table?)
  (for ([key (in-list '("epoch" "batches_that_stepped" "eval_batches"))])
    (check! (fit-loop-key key) exact-nonnegative-integer?))
  (check! (fit-loop-key "epoch_end_logged") boolean?)
  (check-training-state! check! path tree)
  (match (check! (fit-loop-key "phase") string?)
    [(or "epoch-end" "fit-end") (void)]
    [(and phase (or "batch-end" "validation-end" "exception"))
     (not-implemented (format "fit: resuming from a mid-epoch (~a) checkpoint" phase) 'T3)]
    [phase
     (raise-arguments-error
      'fit (string-append "the checkpoint was saved outside an epoch end, a batch end, a validation"
                          " end, an exception and the fit's end, so it cannot be resumed")
      "phase" phase
      "path" path)]))

(define (resumed-epoch tree)
  (match-define (hash-table ("phase" phase) ("epoch" saved)) (fit-loop tree))
  (if (equal? phase "epoch-end") (add1 saved) saved))

(define (check-resume-epoch! ckpt max-epochs)
  (match-define (source path tree) ckpt)
  (define epoch (resumed-epoch tree))
  (when (and max-epochs (> epoch max-epochs))
    (raise-arguments-error
     'fit (format "the checkpoint resumes at epoch ~a, past #:max-epochs ~a" epoch max-epochs)
     "path" path)))

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

(define (path-noted e path)
  (define-values (type skipped?) (struct-info e))
  (match (and type (not skipped?) (struct->vector e))
    [(vector _ message marks fields ...)
     (define noted (format "~a\n  path: ~e" message path))
     (apply (struct-type-make-constructor type) noted marks fields)]
    [#f e]))

(define (call-with-path-noted path thunk)
  (with-handlers ([exn:fail? (lambda (e) (raise (path-noted e path)))])
    (thunk)))

(define-syntax-parse-rule (with-path-noted path:expr body:expr ...+)
  (call-with-path-noted path (lambda () body ...)))

(define (call-with-hparams-put-back-on-escape tr thunk)
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
  (call-with-hparams-put-back-on-escape
   tr
   (lambda ()
     (restore-hparams! r tree)
     (call-trainer r 'on-load-checkpoint tree)
     (with-path-noted path (load-state! tr path #:rename model-entry))))
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
    (with-path-noted path (load! target state))))

(define (warn-inexact-resume! path)
  (~> (string-append "fit: the checkpoint ~a was saved before or during the trainer's"
                     " on-train-epoch-end; values logged there after the save are not in the"
                     " file, so the resume may not be exact; for an exact resume save from a"
                     " #:kind checkpoint callback listed after every callback that logs in"
                     " on-train-epoch-end")
      (format path)
      warn-train!))

(define (restore-training-state! r ckpt)
  (match-define (source path tree) ckpt)
  (define scaler (run-grad-scaler r))
  (define scaler-state (hash-ref tree "MixedPrecision" #f))
  (when (and scaler scaler-state)
    (with-path-noted path (load-grad-scaler-state! scaler scaler-state)))
  (define loop (fit-loop tree))
  (match-define (hash-table ("phase" phase) ("batches_that_stepped" stepped)
                            ("eval_batches" evals))
    loop)
  (unless (hash-ref loop "epoch_end_logged" #t)
    (warn-inexact-resume! path))
  (set-run-epoch! r (resumed-epoch tree))
  (set-run-global-step! r (hash-ref tree "global_step"))
  (set-run-batches-stepped! r stepped)
  (set-run-eval-batches! r evals)
  (with-path-noted path
    (restore-book-state! (run-book r)
                         (hash-ref tree "callback_metrics")
                         (hash-ref tree "metric_book" #f)
                         #:close-train-epoch? (equal? phase "epoch-end")))
  (define tr (run-trainer r))
  (restore-each! path tree "optimizer_states" "optimizer" (run-optimizers r)
                 (lambda (opt state) (load-optimizer-state! opt tr state)))
  (restore-each! path tree "lr_schedulers" "scheduler"
                 (map scheduler-config-scheduler (run-scheduler-configs r))
                 load-scheduler-state!))

(define (restore-rng! r ckpt)
  (match-define (source path tree) ckpt)
  (with-path-noted path (restore-rng-bundle! (hash-ref tree "rng") (run-device r))))
