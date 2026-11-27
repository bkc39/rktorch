#lang racket/base

(require (only-in racket/contract/base
                  -> ->* any/c flat-named-contract hash/c or/c unconstrained-domain->)
         (only-in racket/list partition take)
         (only-in racket/match match match*)
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
                  fresh-run run? run-batches-stepped run-book run-callbacks
                  run-current-stage run-device run-entry run-epoch run-eval-batches
                  run-finished? run-global-step run-grad-scaler run-hook run-optimizers
                  run-scheduler-configs run-train-batch-index run-trainer set-run-finished?!)
         (only-in "private/types.rkt"
                  callback-hook callback-state-key callback-type define-named-contract
                  scheduler-config-scheduler trainer-hook trainer-hparam-names trainer?)
         (only-in "private/value.rkt" checkpoint-value? value-problem))

(define-named-contract checkpoint-value/c checkpoint-value?)

(define-logger rktorch-train)

(define saving? (make-parameter #f))

(define (phase r)
  (match (and (eq? (run-entry r) 'fit) (not (eq? (run-current-stage r) 'sanity-check))
              (match (run-finished? r)
                ['success 'on-fit-end]
                [#f (run-hook r)]
                [_ #f]))
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
    [((or "batch-end" "validation-end" "exception") (? exact-integer? i)) (hash-set counters "batch_index" i)]
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

(define (entry tree key ok? path #:default [default #f])
  (match (hash-ref tree key default)
    [(? ok? v) v]
    [_ (raise-arguments-error 'load-from-checkpoint "the checkpoint's entry is missing or malformed"
                              "key" key
                              "path" path)]))

(define (string-keyed? v)
  (and (hash? v) (for/and ([k (in-hash-keys v)]) (string? k))))

(define (formals? v)
  (and (list? v)
       (for/and ([f (in-list v)])
         (match f
           [(list (or "positional" "rest") (? string?)) #t]
           [(list "keyword" (regexp #rx"^#:.") (? string?)) #t]
           [_ #f]))))

(define (keyword-name kw)
  (string->symbol (keyword->string kw)))

(define (positional-range make-trainer)
  (define mask (procedure-arity-mask make-trainer))
  (values (sub1 (integer-length (bitwise-and mask (- mask))))
          (if (negative? mask) +inf.0 (sub1 (integer-length mask)))))

(define (init-call make-trainer tree overrides path)
  (define saved (entry tree "hyper_parameters" string-keyed? path #:default (hash)))
  (define (missing formal)
    (raise-arguments-error 'load-from-checkpoint
                           "the checkpoint did not save this init argument; pass it in #:init-args"
                           "formal" formal
                           "path" path))
  (define (value-of id)
    (define name (string->symbol id))
    (cond
      [(hash-has-key? overrides name) (hash-ref overrides name)]
      [(hash-has-key? saved id) (hash-ref saved id)]
      [else (missing name)]))
  (define (rest-values id)
    (match (value-of id)
      [(? list? vs) vs]
      [v (raise-arguments-error 'load-from-checkpoint "a rest formal's value must be a list"
                                "formal" (string->symbol id)
                                "value" v)]))
  (define-values (required accepted) (procedure-keywords make-trainer))
  (define-values (fewest most) (positional-range make-trainer))
  (define-values (keywords positional dropped)
    (for/fold ([keywords '()] [positional '()] [dropped '()]
               #:result (values keywords positional (reverse dropped)))
              ([f (in-list (entry tree "init_formals" formals? path))])
      (define room (- most (length positional)))
      (match f
        [(list "keyword" (and text (regexp #rx"^#:(.*)$" (list _ name))) id)
         (define kw (string->keyword name))
         (if (or (not accepted) (memq kw accepted))
             (values (cons (cons kw (value-of id)) keywords) positional dropped)
             (values keywords positional (cons text dropped)))]
        [(list "positional" id)
         (if (positive? room)
             (values keywords (append positional (list (value-of id))) dropped)
             (values keywords positional (cons id dropped)))]
        [(list "rest" id)
         (define vs (if (positive? room) (rest-values id) (hash-ref saved id #f)))
         (cond
           [(and (list? vs) (<= (length vs) room)) (values keywords (append positional vs) dropped)]
           [(positive? room)
            (values keywords (append positional (take vs room)) (cons id dropped))]
           [else (values keywords positional (cons id dropped))])])))
  (define given
    (for/list ([kw (in-list (or accepted required))]
               #:unless (assq kw keywords)
               #:when (or (memq kw required) (hash-has-key? overrides (keyword-name kw))))
      (cons kw (hash-ref overrides (keyword-name kw) (lambda () (missing kw))))))
  (when (< (length positional) fewest)
    (raise-arguments-error 'load-from-checkpoint
                           "the constructor takes more positional arguments than the checkpoint saved"
                           "required" fewest
                           "saved" (length positional)
                           "path" path))
  (unless (null? dropped)
    (log-rktorch-train-warning "load-from-checkpoint: the constructor takes no ~a saved in ~a; dropped"
                               (string-join dropped ", ") path))
  (define sorted (sort (append keywords given) keyword<? #:key car))
  (values (map car sorted) (map cdr sorted) positional))

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
  (restore-hparams! tr (entry tree "trainer_hparams" string-keyed? path #:default (hash)) path)
  (define (counter key) (entry tree key exact-nonnegative-integer? path #:default 0))
  (define r (fresh-run tr #f (trainer-device tr)
                       #:epoch (counter "epoch")
                       #:global-step (counter "global_step")))
  (call-trainer r 'on-load-checkpoint tree)
  (set-run-finished?! r #t)
  (load-state! tr path #:strict? strict?
               #:rename (lambda (name)
                          (and (regexp-match? #rx"^state_dict[.]" name)
                               (substring name (string-length "state_dict.")))))
  (when device
    (to tr device))
  tr)
