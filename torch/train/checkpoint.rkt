#lang racket/base

(require (only-in racket/contract/base
                  -> ->* any/c flat-named-contract hash/c or/c unconstrained-domain->)
         (only-in racket/list filter-map partition split-at)
         (only-in racket/match match match* match-lambda)
         (only-in racket/string string-join)
         (only-in "../foreign.rkt" device? to)
         (only-in "../nn/grad-scaler.rkt" grad-scaler-state-dict)
         (only-in "../nn/optim-state.rkt" optimizer-state-dict)
         (only-in "../nn/scheduler.rkt" scheduler-state-dict)
         (only-in "../nn/state-dict.rkt" load-state! state-dict)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "optimization.rkt" set-hparam! trainer-hparams)
         (only-in "private/book.rkt" book-metrics-state book-state)
         (only-in "private/checks.rkt"
                  formal-name init-capture-kept init-capture-shapes)
         (only-in "private/dispatch.rkt"
                  after-trainer-hook? call-callback call-hooks call-trainer note-epoch-end-save!
                  trainer-device)
         (only-in "private/format.rkt" read-checkpoint-file write-checkpoint!)
         (only-in "private/restore.rkt" rng-bundle)
         (only-in "private/state.rkt"
                  fresh-run run? run-batches-stepped run-book run-callbacks
                  run-current-stage run-device run-entry run-epoch run-eval-batches
                  run-global-step run-grad-scaler run-hook run-optimizers run-outcome
                  run-scheduler-configs run-train-batch-index run-trainer set-run-finished?!
                  set-run-outcome!)
         (only-in "private/types.rkt"
                  callback-hook callback-state-key callback-type define-named-contract
                  scheduler-config-scheduler trainer-hook trainer-hparam-names trainer?)
         (only-in "private/value.rkt"
                  checkpoint-value? string-keyed-state string-keyed? value-problem))

(define-named-contract checkpoint-value/c checkpoint-value?)

(define-logger rktorch-train)

(define saving? (make-parameter #f))

(define (saved-hook r)
  (match (run-outcome r)
    [#f (run-hook r)]
    ['success 'on-fit-end]
    [(or 'failure 'loaded) #f]))

(define (phase r)
  (match (and (eq? (run-entry r) 'fit) (not (eq? (run-current-stage r) 'sanity-check))
              (saved-hook r))
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
    [((or "batch-end" "validation-end" "exception") (? exact-integer? i))
     (hash-set counters "batch_index" i)]
    [("epoch-end" _) (hash-set counters "epoch_end_logged" (after-trainer-hook?))]
    [(_ _) counters]))

(define (formal-entry shape)
  (match shape
    [(list 'keyword kw id)
     (list "keyword" (string-append "#:" (keyword->string kw)) (symbol->string id))]
    [(list kind id) (list (symbol->string kind) (symbol->string id))]))

(define (init-entries tr)
  (define captured ((trainer-hook tr 'init-arguments) tr))
  (define kept (init-capture-kept captured))
  (define shapes (init-capture-shapes captured))
  (hash "hyper_parameters" (string-keyed-state kept "hyper_parameters")
        "init_formals" (map formal-entry shapes)
        "init_dropped" (for/list ([shape (in-list shapes)]
                                  #:unless (hash-has-key? kept (formal-name shape)))
                         (symbol->string (formal-name shape)))))

(define (refused message key-path part . fields)
  (apply raise-arguments-error 'save-checkpoint! message
         (append fields (list "key path" key-path "value" part))))

(define (check-value! v message . fields)
  (match (value-problem v)
    [(cons key-path part) (apply refused message key-path part fields)]
    [_ (void)]))

(define (callback-states r)
  (for/hash ([cb (in-list (run-callbacks r))] #:when (callback-hook cb 'state-dict))
    (define state (call-callback r cb 'state-dict '()))
    (check-value! state "a callback's state-dict is not a checkpoint-value/c"
                  "callback" (callback-type cb)
                  "state key" (callback-state-key cb))
    (values (callback-state-key cb) state)))

(define (model-state r)
  (define tr (run-trainer r))
  (hash-set* (init-entries tr)
             "epoch" (run-epoch r)
             "global_step" (run-global-step r)
             "state_dict" (make-immutable-hash (state-dict tr))
             "loops" (hash "fit_loop" (fit-loop r))
             "trainer_hparams" (string-keyed-state (trainer-hparams tr) "trainer_hparams")
             "callback_metrics" (book-metrics-state (run-book r))))

(define (with-training-state r tree)
  (note-epoch-end-save! r)
  (define full
    (hash-set* tree
               "rng" (rng-bundle (run-device r))
               "callbacks" (callback-states r)
               "metric_book" (book-state (run-book r))
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
  (check-value! tree "a value in the checkpoint is not a checkpoint-value/c"))

(define started-run/c
  (flat-named-contract 'started-run/c (lambda (r) (and (run? r) (run-entry r) #t))))

(define/contract-out (save-checkpoint! r path #:weights-only? [weights-only? #f]) ;; noqa
  (->* (started-run/c path-string?) (#:weights-only? boolean?) void?)
  (when (saving?)
    (raise-arguments-error 'save-checkpoint! "called while a checkpoint is being saved"
                           "path" path))
  (parameterize ([saving? #t])
    (define-values (built on-save)
      (if weights-only?
          (values (model-state r) call-trainer)
          (values (with-training-state r (model-state r)) call-hooks)))
    (define tree (hash-copy built))
    (on-save r 'on-save-checkpoint tree)
    (check-tree! tree)
    (write-checkpoint! 'save-checkpoint! path tree)))

(define/contract-out (read-checkpoint path) ;; noqa
  (-> path-string? (hash/c string? checkpoint-value/c #:immutable #t))
  (read-checkpoint-file 'read-checkpoint path))

(define (checkpoint-entry tree key ok? path #:default [default #f])
  (match (hash-ref tree key default)
    [(? ok? v) v]
    [_ (raise-arguments-error 'load-from-checkpoint "the checkpoint's entry is missing or malformed"
                              "key" key
                              "path" path)]))

(define (formals? v)
  (and (list? v)
       (for/and ([f (in-list v)])
         (match f
           [(list (or "positional" "rest") (? string?)) #t]
           [(list "keyword" (regexp #rx"^#:.") (? string?)) #t]
           [_ #f]))))

(define (keyword-name kw)
  (string->symbol (keyword->string kw)))

(define (saved-formals tree path)
  (for/fold ([keywords '()] [positionals '()] [rest #f]
             #:result (values (reverse keywords) (reverse positionals) rest))
            ([f (in-list (checkpoint-entry tree "init_formals" formals? path))])
    (match f
      [(list "keyword" (and text (regexp #rx"^#:(.*)$" (list _ (app string->keyword kw)))) id)
       (values (cons (list kw text id) keywords) positionals rest)]
      [(list "positional" id) (values keywords (cons id positionals) rest)]
      [(list "rest" id) (values keywords positionals id)])))

(define (positional-limit proc)
  (let most ([arity (procedure-arity proc)])
    (match arity
      [(? exact-nonnegative-integer? n) n]
      [(arity-at-least _) #f]
      [(list arities ...)
       (define limits (map most arities))
       (and (andmap values limits) (apply max 0 limits))])))

(define (missing-init-arg formal path)
  (raise-arguments-error 'load-from-checkpoint
                         "the checkpoint did not save this init argument; pass it in #:init-args"
                         "formal" formal
                         "path" path))

(define ((init-value saved overrides path) id)
  (define name (string->symbol id))
  (cond
    [(hash-has-key? overrides name) (hash-ref overrides name)]
    [(hash-has-key? saved id) (hash-ref saved id)]
    [else (missing-init-arg name path)]))

(define (rest-values value-of id)
  (match (value-of id)
    [(? list? vs) vs]
    [v (raise-arguments-error 'load-from-checkpoint "a rest formal's value must be a list"
                              "formal" (string->symbol id)
                              "value" v)]))

(define (rest-passed? limit overrides rest-id)
  (and rest-id (not limit) (hash-has-key? overrides (string->symbol rest-id))))

(define (positional-args make-trainer limit value-of overrides positional-ids rest-id path)
  (define saved-count (length positional-ids))
  (define-values (passed-ids extra-ids)
    (split-at positional-ids (if limit (min limit saved-count) saved-count)))
  (define given
    (if (rest-passed? limit overrides rest-id) (rest-values value-of rest-id) '()))
  (define positional (append (map value-of passed-ids) given))
  (unless (procedure-arity-includes? make-trainer (length positional) #t)
    (apply raise-arguments-error
           'load-from-checkpoint
           (string-append "the constructor does not accept the number of positional arguments"
                          " it would be passed; wrap it in a procedure that takes them")
           "saved" saved-count
           (append (if (null? given) '() (list "given" (length given)))
                   (list "accepts" (procedure-arity make-trainer)
                         "path" path))))
  (values positional extra-ids))

(define (keyword-args make-trainer value-of overrides keyword-formals path)
  (define-values (required accepted) (procedure-keywords make-trainer))
  (define-values (taken untaken)
    (partition (match-lambda [(list kw _ _) (or (not accepted) (memq kw accepted))])
               keyword-formals))
  (define recorded (map (match-lambda [(list kw _ id) (cons kw (value-of id))]) taken))
  (define given
    (for/list ([kw (in-list (or accepted required))]
               #:unless (assq kw recorded)
               #:when (or (memq kw required) (hash-has-key? overrides (keyword-name kw))))
      (cons kw (hash-ref overrides (keyword-name kw) (lambda () (missing-init-arg kw path))))))
  (define sorted (sort (append recorded given) keyword<? #:key car))
  (values (map car sorted) (map cdr sorted) untaken))

(define (warn-dropped! limit saved overrides path extra-ids untaken-keywords rest-id)
  (define (saved? id) (hash-has-key? saved id))
  (define untaken
    (append (filter-map (lambda (id) (and (saved? id) (string-append "positional " id)))
                        extra-ids)
            (filter-map (match-lambda
                          [(list _ text id) (and (saved? id) (string-append "keyword " text))])
                        untaken-keywords)))
  (define clauses
    (append (if (null? untaken)
                '()
                (list (string-join untaken ", " #:before-first "the constructor takes no ")))
            (if (and rest-id
                     (not (rest-passed? limit overrides rest-id))
                     (pair? (hash-ref saved rest-id '())))
                (list (string-append "saved rest values for " rest-id " are never passed"))
                '())))
  (unless (null? clauses)
    (log-rktorch-train-warning "load-from-checkpoint: dropped init arguments saved in ~a: ~a"
                               path
                               (string-join clauses "; "))))

(define (init-call make-trainer tree overrides path)
  (define saved (checkpoint-entry tree "hyper_parameters" string-keyed? path #:default (hash)))
  (define value-of (init-value saved overrides path))
  (define-values (keyword-formals positional-ids rest-id) (saved-formals tree path))
  (define limit (positional-limit make-trainer))
  (define-values (positional extra-ids)
    (positional-args make-trainer limit value-of overrides positional-ids rest-id path))
  (define-values (kws kw-values untaken-keywords)
    (keyword-args make-trainer value-of overrides keyword-formals path))
  (warn-dropped! limit saved overrides path extra-ids untaken-keywords rest-id)
  (values kws kw-values positional))

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
  (define-values (kws kw-values positional) (init-call make-trainer tree init-args path))
  (define tr (keyword-apply make-trainer kws kw-values positional))
  (restore-hparams! tr
                    (checkpoint-entry tree "trainer_hparams" string-keyed? path #:default (hash))
                    path)
  (define (counter key) (checkpoint-entry tree key exact-nonnegative-integer? path #:default 0))
  (define r (fresh-run tr #f (trainer-device tr)
                       #:epoch (counter "epoch")
                       #:global-step (counter "global_step")))
  (call-trainer r 'on-load-checkpoint tree)
  (set-run-finished?! r #t)
  (set-run-outcome! r 'loaded)
  (load-state! tr path #:strict? strict?
               #:rename (lambda (name)
                          (and (regexp-match? #rx"^state_dict[.]" name)
                               (substring name (string-length "state_dict.")))))
  (when device
    (to tr device))
  tr)
