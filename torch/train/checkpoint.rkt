#lang racket/base

(require (only-in racket/contract/base
                  -> ->* any/c flat-named-contract hash/c or/c unconstrained-domain->)
         (only-in racket/list append* partition)
         (only-in racket/match match match* match-define)
         (only-in racket/string string-join)
         (only-in "../foreign.rkt" device? to)
         (only-in "../nn/grad-scaler.rkt" grad-scaler-state-dict)
         (only-in "../nn/optim-state.rkt" optimizer-state-dict)
         (only-in "../nn/scheduler.rkt" scheduler-state-dict)
         (only-in "../nn/state-dict.rkt" load-state! state-dict)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "optimization.rkt" set-hparam! trainer-hparams)
         (only-in "private/book.rkt" book-metrics)
         (only-in "private/checks.rkt"
                  formal-name init-capture-kept init-capture-shapes)
         (only-in "private/dispatch.rkt" call-callback call-hooks call-trainer trainer-device)
         (only-in "private/format.rkt" read-checkpoint-file write-checkpoint!)
         (only-in "private/restore.rkt" rng-bundle)
         (only-in "private/state.rkt"
                  fresh-run live? run-batches-stepped run-book run-callbacks
                  run-current-stage run-device run-entry run-epoch run-eval-batches
                  run-global-step run-grad-scaler run-hook run-optimizers
                  run-scheduler-configs run-train-batch-index run-trainer set-run-finished?!)
         (only-in "private/types.rkt"
                  callback-hook callback-state-key callback-type define-named-contract
                  scheduler-config-scheduler trainer-hook trainer-hparam-names trainer?)
         (only-in "private/value.rkt" checkpoint-value? value-problem)
         (only-in "run.rkt" run?))

(define-named-contract checkpoint-value/c checkpoint-value?)

(define-logger rktorch-train)

(define saving? (make-parameter #f))

(define (phase r)
  (match (and (eq? (run-entry r) 'fit) (not (eq? (run-current-stage r) 'sanity-check))
              (run-hook r))
    ['on-train-epoch-end "epoch-end"]
    ['on-train-batch-end "batch-end"]
    ['on-validation-end "validation-end"]
    ['on-exception "exception"]
    [(or 'on-train-end 'on-fit-end) "fit-end"]
    [_ "other"]))

(define (fit-loop r)
  (define at (phase r))
  (define counters
    (hash "epoch" (run-epoch r)
          "phase" at
          "batches_that_stepped" (run-batches-stepped r)
          "eval_batches" (run-eval-batches r)))
  (match* (at (run-train-batch-index r))
    [((or "batch-end" "validation-end") (? exact-integer? i)) (hash-set counters "batch_index" i)]
    [(_ _) counters]))

(define (string-keyed h)
  (for/hash ([(k v) (in-hash h)]) (values (symbol->string k) v)))

(define (formal-entry shape)
  (match shape
    [(list 'keyword kw id)
     (list "keyword" (string-append "#:" (keyword->string kw)) (symbol->string id))]
    [(list kind id) (list (symbol->string kind) (symbol->string id))]))

(define (init-entries tr)
  (define captured ((trainer-hook tr 'init-arguments) tr))
  (define kept (init-capture-kept captured))
  (define shapes (init-capture-shapes captured))
  (hash "hyper_parameters" (string-keyed kept)
        "init_formals" (map formal-entry shapes)
        "init_dropped" (for/list ([shape (in-list shapes)]
                                  #:unless (hash-has-key? kept (formal-name shape)))
                         (symbol->string (formal-name shape)))))

(define (refused message key-path part . fields)
  (apply raise-arguments-error 'save-checkpoint! message
         (append fields (list "key path" key-path "value" part))))

(define (callback-states r)
  (for/hash ([cb (in-list (run-callbacks r))] #:when (callback-hook cb 'state-dict))
    (define state (call-callback r cb 'state-dict '()))
    (match (value-problem state)
      [(cons path part)
       (refused "a callback's state-dict is not a checkpoint-value/c" path part
                "callback" (callback-type cb)
                "state key" (callback-state-key cb))]
      [_ (void)])
    (values (callback-state-key cb) state)))

(define (model-state r)
  (define tr (run-trainer r))
  (hash-set* (init-entries tr)
             "epoch" (run-epoch r)
             "global_step" (run-global-step r)
             "state_dict" (make-immutable-hash (state-dict tr))
             "loops" (hash "fit_loop" (fit-loop r))
             "trainer_hparams" (string-keyed (trainer-hparams tr))
             "callback_metrics" (string-keyed (book-metrics (run-book r)))))

(define (with-training-state r tree)
  (define full
    (hash-set* tree
               "rng" (rng-bundle (run-device r))
               "callbacks" (callback-states r)
               "optimizer_states" (for/list ([opt (in-list (run-optimizers r))])
                                    (optimizer-state-dict opt (run-trainer r)))
               "lr_schedulers" (for/list ([c (in-list (run-scheduler-configs r))])
                                 (scheduler-state-dict (scheduler-config-scheduler c)))))
  (match (run-grad-scaler r)
    [#f full]
    [s (hash-set full "MixedPrecision" (grad-scaler-state-dict s))]))

(define (check-tree! tree)
  (for ([k (in-hash-keys tree)] #:unless (string? k))
    (refused "a top-level key of the checkpoint is not a string" (list k) k))
  (match (value-problem tree)
    [(cons key-path part) (refused "a value in the checkpoint is not a checkpoint-value/c" key-path part)]
    [_ (void)]))

(define active-run/c
  (flat-named-contract 'active-run/c (lambda (r) (and (live? r) (run-entry r) #t))))

(define/contract-out (save-checkpoint! r path #:weights-only? [weights-only? #f]) ;; noqa
  (->* (active-run/c path-string?) (#:weights-only? boolean?) void?)
  (when (saving?)
    (raise-arguments-error 'save-checkpoint! "called while a checkpoint is being saved"
                           "path" path))
  (parameterize ([saving? #t])
    (define tree
      (hash-copy (if weights-only? (model-state r) (with-training-state r (model-state r)))))
    (if weights-only?
        (call-trainer r 'on-save-checkpoint tree)
        (call-hooks r 'on-save-checkpoint tree))
    (check-tree! tree)
    (write-checkpoint! 'save-checkpoint! path tree)))

(define/contract-out (read-checkpoint path) ;; noqa
  (-> path-string? (hash/c string? checkpoint-value/c #:immutable #t))
  (read-checkpoint-file 'read-checkpoint path))

(define (init-call tree overrides)
  (define saved (hash-ref tree "hyper_parameters"))
  (define (value-of id)
    (define name (string->symbol id))
    (cond
      [(hash-has-key? overrides name) (hash-ref overrides name)]
      [(hash-has-key? saved id) (hash-ref saved id)]
      [else (raise-arguments-error
             'load-from-checkpoint
             "the checkpoint did not save this init argument; pass it in #:init-args"
             "formal" name)]))
  (define (rest-values id)
    (match (value-of id)
      [(? list? vs) vs]
      [v (raise-arguments-error 'load-from-checkpoint "a rest formal's value must be a list"
                                "formal" (string->symbol id)
                                "value" v)]))
  (define formals (hash-ref tree "init_formals"))
  (define keywords
    (sort (for/list ([f (in-list formals)] #:when (equal? (car f) "keyword"))
            (match-define (list _ kw id) f)
            (cons (string->keyword (substring kw 2)) (value-of id)))
          keyword<? #:key car))
  (define positional
    (append* (for/list ([f (in-list formals)])
               (match f
                 [(list "positional" id) (list (value-of id))]
                 [(list "rest" id) (rest-values id)]
                 [_ '()]))))
  (values (map car keywords) (map cdr keywords) positional))

(define (restore-hparams! tr saved path)
  (define-values (known unknown)
    (partition (lambda (name) (memq name (trainer-hparam-names tr)))
               (map string->symbol (sort (hash-keys saved) string<?))))
  (for ([name (in-list known)])
    (set-hparam! tr name (hash-ref saved (symbol->string name))))
  (unless (null? unknown)
    (log-rktorch-train-warning
     "load-from-checkpoint: the trainer has no hparam ~a saved in ~a; skipped"
     (string-join (map symbol->string unknown) ", ") path)))

(define/contract-out (load-from-checkpoint make-trainer path ;; noqa
                                           #:device [device #f]
                                           #:strict? [strict? #t]
                                           #:init-args [init-args (hasheq)])
  (->* ((unconstrained-domain-> trainer?) path-string?)
       (#:device (or/c #f device?)
        #:strict? boolean?
        #:init-args (hash/c symbol? any/c #:immutable #t))
       trainer?)
  (define tree (read-checkpoint-file 'load-from-checkpoint path))
  (define-values (kws kw-values positional) (init-call tree init-args))
  (define tr (keyword-apply make-trainer kws kw-values positional))
  (restore-hparams! tr (hash-ref tree "trainer_hparams") path)
  (define r (fresh-run tr #f (trainer-device tr)
                       #:epoch (hash-ref tree "epoch")
                       #:global-step (hash-ref tree "global_step")))
  (call-trainer r 'on-load-checkpoint tree)
  (set-run-finished?! r #t)
  (load-state! tr path #:strict? strict?
               #:rename (lambda (name)
                          (and (regexp-match? #rx"^state_dict[.]" name)
                               (substring name (string-length "state_dict.")))))
  (when device
    (to tr device))
  tr)
