#lang racket/base

(require (only-in racket/contract/base
                  -> ->* any/c hash/c or/c unconstrained-domain->)
         (only-in racket/list partition)
         (only-in racket/match match match*)
         (only-in "../foreign.rkt" device? to)
         (only-in "../nn/grad-scaler.rkt" grad-scaler-state-dict)
         (only-in "../nn/optim-state.rkt" optimizer-state-dict)
         (only-in "../nn/scheduler.rkt" scheduler-state-dict)
         (only-in "../nn/state-dict.rkt" load-state! state-dict)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "optimization.rkt" set-hparam! trainer-hparams)
         (only-in "private/book.rkt" book-metrics)
         (only-in "private/checks.rkt" captured-arguments-shapes captured-arguments-values formal-name)
         (only-in "private/dispatch.rkt" call-callback call-hooks call-trainer trainer-device)
         (only-in "private/format.rkt" read-checkpoint-file write-checkpoint!)
         (only-in "private/restore.rkt" rng-bundle)
         (only-in "private/state.rkt"
                  fresh-run run-batch-index run-batches-stepped run-book run-callbacks
                  run-current-stage run-device run-entry run-epoch run-eval-batches
                  run-global-step run-grad-scaler run-hook run-optimizers
                  run-scheduler-configs run-trainer set-run-finished?!)
         (only-in "private/types.rkt"
                  callback-hook callback-state-key callback-type define-named-contract
                  scheduler-config-scheduler trainer-hook trainer?)
         (only-in "private/value.rkt" checkpoint-value? value-problem)
         (only-in "run.rkt" run?))

(define-named-contract checkpoint-value/c checkpoint-value?)

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
  (match* (at (run-batch-index r))
    [((or "batch-end" "validation-end") (? exact-integer? i)) (hash-set counters "batch_index" i)]
    [(_ _) counters]))

(define (string-keyed h)
  (for/hash ([(k v) (in-hash h)]) (values (symbol->string k) v)))

(define (formal-entry shape)
  (match shape
    [(list 'keyword kw id) (list "keyword" (string-append "#:" (keyword->string kw)) (symbol->string id))]
    [(list kind id) (list (symbol->string kind) (symbol->string id))]))

(define (init-entries tr)
  (define captured ((trainer-hook tr 'init-arguments) tr))
  (define shapes (captured-arguments-shapes captured))
  (define-values (kept dropped)
    (partition (lambda (e) (checkpoint-value? (cdr e)))
               (map cons shapes (captured-arguments-values captured))))
  (define (id-of shape) (symbol->string (formal-name shape)))
  (hash "hyper_parameters" (for/hash ([e (in-list kept)]) (values (id-of (car e)) (cdr e)))
        "init_formals" (map formal-entry shapes)
        "init_dropped" (map (lambda (e) (id-of (car e))) dropped)))

(define (refused who what path part)
  (raise-arguments-error who (format "~a is not a checkpoint-value/c" what)
                         "key path" path
                         "value" part))

(define (callback-states r)
  (for/hash ([cb (in-list (run-callbacks r))] #:when (callback-hook cb 'state-dict))
    (define state (call-callback r cb 'state-dict '()))
    (match (value-problem state)
      [#f (void)]
      [(cons path part)
       (raise-arguments-error 'save-checkpoint!
                              "a callback's state-dict is not a checkpoint-value/c"
                              "callback" (callback-type cb)
                              "state key" (callback-state-key cb)
                              "key path" path
                              "value" part)])
    (values (callback-state-key cb) state)))

(define (base-tree r weights-only?)
  (define tr (run-trainer r))
  (define tree
    (make-hash
     (list (cons "epoch" (run-epoch r))
           (cons "global_step" (run-global-step r))
           (cons "state_dict" (make-immutable-hash (state-dict tr)))
           (cons "loops" (hash "fit_loop" (fit-loop r)))
           (cons "trainer_hparams" (string-keyed (trainer-hparams tr)))
           (cons "callback_metrics" (string-keyed (book-metrics (run-book r)))))))
  (for ([(k v) (in-hash (init-entries tr))])
    (hash-set! tree k v))
  (unless weights-only?
    (hash-set! tree "rng" (rng-bundle (run-device r))))
  tree)

(define (add-training-state! r tree)
  (define tr (run-trainer r))
  (hash-set! tree "callbacks" (callback-states r))
  (hash-set! tree "optimizer_states"
             (for/list ([opt (in-list (run-optimizers r))]) (optimizer-state-dict opt tr)))
  (hash-set! tree "lr_schedulers"
             (for/list ([c (in-list (run-scheduler-configs r))])
               (scheduler-state-dict (scheduler-config-scheduler c))))
  (match (run-grad-scaler r)
    [#f (void)]
    [s (hash-set! tree "MixedPrecision" (grad-scaler-state-dict s))]))

(define/contract-out (save-checkpoint! r path #:weights-only? [weights-only? #f]) ;; noqa
  (->* (run? path-string?) (#:weights-only? boolean?) void?)
  (when (saving?)
    (raise-arguments-error 'save-checkpoint! "called while a checkpoint is being saved"
                           "path" path))
  (parameterize ([saving? #t])
    (define tree (base-tree r weights-only?))
    (cond
      [weights-only? (call-trainer r 'on-save-checkpoint tree)]
      [else
       (add-training-state! r tree)
       (call-hooks r 'on-save-checkpoint tree)])
    (match (value-problem tree)
      [#f (void)]
      [(cons key-path part) (refused 'save-checkpoint! "a value in the checkpoint" key-path part)])
    (write-checkpoint! 'save-checkpoint! path tree)))

(define/contract-out (read-checkpoint path) ;; noqa
  (-> path-string? (hash/c string? checkpoint-value/c #:immutable #t))
  (read-checkpoint-file 'read-checkpoint path))

(define (init-call tree overrides)
  (define saved (hash-ref tree "hyper_parameters"))
  (define formals (hash-ref tree "init_formals"))
  (define (value-of id)
    (define name (string->symbol id))
    (cond
      [(hash-has-key? overrides name) (hash-ref overrides name)]
      [(hash-has-key? saved id) (hash-ref saved id)]
      [else (raise-arguments-error 'load-from-checkpoint
                                   "the checkpoint did not save this init argument; pass it in #:init-args"
                                   "formal" name)]))
  (define ids (map (lambda (f) (string->symbol (match f [(list _ ... id) id]))) formals))
  (for ([name (in-list (sort (hash-keys overrides) symbol<?))])
    (unless (memq name ids)
      (raise-arguments-error 'load-from-checkpoint "#:init-args names no init formal"
                             "name" name
                             "formals" ids)))
  (for/fold ([positional '()] [keywords '()]
             #:result (let ([kws (sort keywords keyword<? #:key car)])
                        (values (map car kws) (map cdr kws) (reverse positional))))
            ([f (in-list formals)])
    (match f
      [(list "keyword" kw id) (values positional
                                      (cons (cons (string->keyword (substring kw 2)) (value-of id))
                                            keywords))]
      [(list "positional" id) (values (cons (value-of id) positional) keywords)]
      [(list "rest" id) (values (append (reverse (value-of id)) positional) keywords)])))

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
  (for ([(name v) (in-hash (hash-ref tree "trainer_hparams"))])
    (set-hparam! tr (string->symbol name) v))
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
