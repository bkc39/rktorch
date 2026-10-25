#lang scribble/manual

@(require (for-syntax racket/base
                      (only-in syntax/parse/pre id this-syntax)
                      (only-in torch/train/private/hooks
                               callback-hook? definable-hooks hook-batch?
                               hook-formals hook-name))
          (only-in racket/string string-join)
          (only-in syntax/parse/define define-syntax-parse-rule)
          "common.rkt"
          torch/scribblings/labels
          (for-label (only-in torch/data/loader dataloader dataloader?
                              tensor-dataset))
          (only-in torch/train/private/hooks
                   epoch-log epoch-only-log hook-batch? hook-form hook-formals
                   hook-log hook-name hook-order hook-sides hooks step-log))

@title[#:tag "train"]{Training}

@defmodule[torch/train]

A trainer is a layer whose body forms are a method's steps and hooks, and
@racket[fit] runs it in Lightning 2.6.5's loop order. This chapter is the
specification: the definers check their syntax, and a procedure that is not
implemented yet raises @racket[exn:fail:unsupported].

@defform[(define-trainer name (field ...) option ... body-form ...)
         #:grammar
         ([option (code:line #:contract contract-expr)
                  (code:line #:predicate id)
                  (code:line #:automatic-optimization? boolean)]
          [body-form (define-init init-formals init-body ...+)
                     (define-forward forward-formals body ...+)
                     (define-optimizers body ...+)
                     (define-callbacks body ...+)
                     (define-hparams field-id ...+)
                     (define-training-step step-formals body ...+)
                     (define-validation-step step-formals body ...+)
                     (define-test-step step-formals body ...+)
                     (define-predict-step step-formals body ...+)
                     (define-hook (hook-id . hook-formals) body ...+)]
          [step-formals (input ...) (input ... . rest-id)]
          [input id [id : input-contract-expr]])]{
Defines a trainer: a @racket[define-layer] layer whose body forms are the
method's steps and hooks, the analogue of a @tt{LightningModule}. Its
parameters, buffers and state dict are its fields', and it moves with
@racket[to]. Every body form is optional and appears at most once,
@racket[define-hook] once per @racket[hook-id]. Bodies see the fields,
@racket[run] and @racket[log!]; @racket[set!] on a field writes the instance.
A field that held a parameter, buffer or layer after @racket[define-init] is
fixed: assigning it raises @racket[exn:fail:contract], and the tensor is
mutated in place instead. A @racket[step-formals] input written
@racket[[id : input-contract-expr]] is checked as @racket[define-layer]
checks a forward input.

@racketblock[
(define-trainer Line (net)
  (define-init () (set! net (Linear 1 1)))
  (define-optimizers (sgd (parameters net) #:lr 0.05))
  (define-training-step (x y)
    (define loss (mse-loss (net x) y))
    (log! 'train/loss loss #:on-epoch? #t)
    loss))
(define xs (tensor '((0.0) (1.0) (2.0) (3.0))))
(define r (fit (Line) (dataloader (tensor-dataset xs (* 2 xs)) #:batch-size 2)
               #:max-epochs 3 #:logger #f))
(run-global-step r)
(real? (run-metric r 'train/loss))
]}

@defform[(define-callback name (field ...) option ... cb-form ...)
         #:grammar
         ([option (code:line #:contract contract-expr)
                  (code:line #:predicate id)
                  (code:line #:kind plain)
                  (code:line #:kind monitor)
                  (code:line #:kind checkpoint)]
          [cb-form (define-init init-formals init-body ...+)
                   (define-state-key expr)
                   (define-hook (hook-id . hook-formals) body ...+)])]{
Defines a callback, Lightning's @tt{Callback}: reusable code that hooks into
the loop. A callback is a layer; @racket[fit] moves it with the trainer.
Callbacks run in list order before the trainer's hook. A @racket[checkpoint]
callback runs after every other callback. @racket[monitor] and
@racket[checkpoint] callbacks run after the trainer's
@racket[on-train-epoch-end], so they see what it logged. Nothing is saved
unless the callback defines the @racket[state-dict] and @racket[load-state!]
hooks.}

@defform[(define-init init-formals init-body ...+)]{
The constructor body, as @racket[define-layer]'s @racket[#:init];
@racket[run] is not bound.}

@defform[(define-forward forward-formals body ...+)]{
The trainer's forward, as @racket[define-layer]'s @racket[#:forward]. Without
it, applying the trainer raises @racket[exn:fail:contract].}

@defform[(define-optimizers body ...+)]{
Lightning's @tt{configure_optimizers}. Called by @racket[fit] after
@racket[setup] and the device move. Returns optimizers, @racket[scheduler?]
values (stepped once per epoch) and @racket[scheduler-config] values.
Automatic optimization takes exactly one optimizer.}

@defform[(define-callbacks body ...+)]{
Lightning's @tt{configure_callbacks}: the callbacks the method cannot work
without, as values or lists. They run before @racket[fit]'s
@racket[#:callbacks], and they replace any of those callbacks of the same
type.}

@defform[(define-hparams field-id ...+)]{
Names the fields that hold schedulable real hyperparameters.
@racket[set-hparam!] sets them, the logger receives them at fit start, and
checkpoints carry them.}

@deftogether[(@defform[(define-training-step step-formals body ...+)]
              @defform[(define-validation-step step-formals body ...+)]
              @defform[(define-test-step step-formals body ...+)]
              @defform[(define-predict-step step-formals body ...+)])]{
The steps; @racket[step-formals] receive the values of one batch. The
training step answers a loss (a floating @racket[scalar-tensor/c]),
@racket[(values loss extra)], or @racket[#f], which skips the step as
@racket[skip-step!] does. The other steps answer one value, which reaches
their batch-end hook. The predict step's values are collected by
@racket[predict].}

@defform[(define-hook (hook-id . hook-formals) body ...+)]{
Defines the hook @racket[hook-id] from @secref["train-hooks"]. The name, the
side and the arity are checked at compile time. @racket[state-dict] and
@racket[load-state!] name hooks here; in the body those identifiers keep
their @racketmodname[torch/nn] meaning.
@examples[#:eval train-eval
(eval:error (define-callback typo () (define-hook (on-batch-end) (void))))]}

@defform[(define-state-key expr)]{
The callback's key in a checkpoint (a @racket[state-key/c]). The default is
the callback's name.}

@defidform[run]{Bound in hook and step bodies to the current @racket[run?].}

@defform[(log! key value option ...)]{
@racket[(run-log! run key value option ...)]; a bare @racket[log!] is that
procedure.}

@section{Running}

@defproc[(fit [trainer trainer?]
              [train dataloader?]
              [#:val val (or/c #f dataloader?) #f]
              [#:max-epochs max-epochs (or/c #f exact-positive-integer?) #f]
              [#:max-steps max-steps (or/c #f exact-positive-integer?) #f]
              [#:device device (or/c #f device?) #f]
              [#:logger logger logger-spec/c (stdout-logger)]
              [#:callbacks callbacks (listof callback?) '()]
              [#:accumulate-grad-batches accumulate exact-positive-integer? 1]
              [#:gradient-clip-val clip-val (or/c #f (>/c 0)) #f]
              [#:gradient-clip-algorithm clip-algorithm clip-algorithm/c 'norm]
              [#:precision precision precision/c '32-true]
              [#:check-val-every-n-epoch val-every exact-positive-integer? 1]
              [#:val-check-interval val-interval val-check-interval/c #f]
              [#:num-sanity-val-steps sanity exact-nonnegative-integer? 2]
              [#:limit-train-batches limit-train limit/c 1.0]
              [#:limit-val-batches limit-val limit/c 1.0]
              [#:log-every-n-steps log-every exact-positive-integer? 50]
              [#:fast-dev-run fast-dev-run fast-dev-run/c #f]
              [#:ckpt-path ckpt-path (or/c #f path-string?) #f])
         run?]{
Trains @racket[trainer] in the order of @secref["train-hooks"] and answers
the finished run. Keywords follow Lightning's @tt{Trainer}. A
@racket[#:device] of @racket[#f] leaves the trainer where it is and sends
each batch to the trainer's device. @racket[log!] reduces on the device and
reads to the host at every @racket[#:log-every-n-steps]-th optimizer step of
the fit, counted across epochs, and at each epoch end, in one transfer for
all keys. @racket[#:fast-dev-run] runs that many batches (one for
@racket[#t]) of one epoch, with no sanity check, no loggers and no
@racket[monitor] or @racket[checkpoint] callbacks. If an @racket[exn:fail]
or @racket[exn:break] is raised, @racket[fit] runs each callback's
@racket[on-exception] and then @racket[teardown] with breaks disabled, and
re-raises the exception; a break issued meanwhile is absorbed.

The contract also requires @racket[#:max-epochs], @racket[#:max-steps] or
@racket[#:fast-dev-run]; a @racket[define-training-step]; a
@racket[define-optimizers] under automatic optimization; a
@racket[define-validation-step] with @racket[#:val]; limits that select at
least one batch; an integer @racket[#:val-check-interval] no larger than the
number of training batches @racket[#:limit-train-batches] selects; and, under
manual optimization, no
@racket[#:accumulate-grad-batches] above 1 and no
@racket[#:gradient-clip-val].}

@defproc[(test [trainer trainer?]
               [loader dataloader?]
               [#:device device (or/c #f device?) #f]
               [#:logger logger logger-spec/c (stdout-logger)]
               [#:callbacks callbacks (listof callback?) '()]
               [#:limit-test-batches limit limit/c 1.0]
               [#:ckpt-path ckpt-path (or/c #f path-string?) #f])
         run?]{
The test loop: @racket[setup] @racket['test], the @racket[on-test-*] hooks,
@racket[teardown]. The trainer needs a @racket[define-test-step].}

@defproc[(predict [trainer trainer?]
                  [loader dataloader?]
                  [#:device device (or/c #f device?) #f]
                  [#:callbacks callbacks (listof callback?) '()]
                  [#:limit-predict-batches limit limit/c 1.0]
                  [#:ckpt-path ckpt-path (or/c #f path-string?) #f])
         list?]{
The predict loop; answers the predict step's output per batch. Nothing is
logged. The trainer needs a @racket[define-predict-step].}

@deftogether[(@defproc[(trainer? [v any/c]) boolean?]
              @defproc[(callback? [v any/c]) boolean?])]{
Recognize the instances of @racket[define-trainer] and
@racket[define-callback] types.}

@section{The run}

@deftogether[(@defproc[(run? [v any/c]) boolean?]
              @defproc[(run-trainer [r run?]) trainer?]
              @defproc[(run-stage [r run?]) (or/c #f run-stage/c)]
              @defproc[(run-epoch [r run?]) exact-nonnegative-integer?]
              @defproc[(run-global-step [r run?]) exact-nonnegative-integer?]
              @defproc[(run-batch-index [r run?])
                       (or/c #f exact-nonnegative-integer?)]
              @defproc[(run-optimizers [r run?]) (listof optimizer?)]
              @defproc[(run-device [r run?]) device?]
              @defproc[(run-stepped? [r run?]) boolean?]
              @defproc[(run-metric [r run?] [key symbol?]
                                   [#:default default any/c #f])
                       any]
              @defproc[(run-metrics [r run?])
                       (hash/c symbol? real? #:immutable #t)])]{
The loop state. @racket[run-stage] is @racket[#f] once the run has finished.
@racket[run-epoch] is Lightning's @tt{current_epoch}.
@racket[run-global-step] counts optimizer steps taken.
@racket[run-stepped?] is whether the current training batch took one.
@racket[run-metric] reads one key to the host: inside an epoch the latest
step value, after an epoch its reduction, and for a key logged both per step
and per epoch the newer of the two.}

@deftogether[(@defproc[(run-schedulers [r run?]) (listof scheduler-config?)]
              @defproc[(run-estimated-stepping-batches [r run?])
                       exact-nonnegative-integer?])]{
The schedulers @racket[define-optimizers] returned, and Lightning's
@tt{estimated_stepping_batches}.}

@defproc[(request-stop! [r fitting-run/c]) void?]{
Lightning's @tt{should_stop}: the current batch finishes, then the
validation loop and the epoch-end and fit-end hooks run.}

@defproc[(skip-step! [r step-pending-run/c]) void?]{
Drops the pending optimizer step. The window's remaining backward passes do
not run, no step is taken, and @racket[run-global-step] does not advance.}

@defproc[(skip-epoch! [r batch-start-run/c]) void?]{
Lightning's @racket[-1] from @tt{on_train_batch_start}. Once every
@racket[on-train-batch-start] hook of the batch has run, the rest of the
epoch is skipped, its validation loop included.}

@section{Logging}

@defproc[(run-log! [r logging-run/c]
                   [key symbol?]
                   [value metric-value/c]
                   [#:progress? progress? boolean? #f]
                   [#:on-step? on-step? boolean? @#,elem{the hook's default}]
                   [#:on-epoch? on-epoch? boolean? @#,elem{the hook's default}]
                   [#:reduce reduce reduce/c 'mean]
                   [#:batch-size batch-size (or/c #f exact-positive-integer?) #f]
                   [#:logger? logger? boolean? #t])
         void?]{
Lightning's @tt{self.log}. The values allowed for @racket[#:on-step?] and
@racket[#:on-epoch?], and their defaults, are the hook's, from
@secref["train-hooks"]; @racket[#:on-step? #t] or @racket[#:on-epoch? #f]
in a hook that logs only per epoch is a contract violation. An epoch
@racket['mean] is weighted by @racket[#:batch-size].}

@defthing[gen:logger any/c]{
The generic interface of a logger, with the methods below.}

@deftogether[(@defproc[(logger? [v any/c]) boolean?]
              @defproc[(logger-log-hyperparams! [logger logger?]
                                                [params (hash/c symbol? any/c)])
                       any]
              @defproc[(logger-log-metrics! [logger logger?]
                                            [metrics (hash/c symbol? real?)]
                                            [step exact-nonnegative-integer?])
                       any]
              @defproc[(logger-finalize! [logger logger?]
                                         [status (or/c 'success 'failed)])
                       any])]{
A logger receives the hyperparameters once at fit start, the host metrics at
each read, and @racket['failed] when the entry point raises.}

@defproc[(stdout-logger [#:out out output-port? (current-output-port)])
         logger?]{Prints each metrics read as one line.}

@deftogether[(@defproc[(jsonl-logger [dir path-string?]
                                     [#:exists exists
                                      (or/c 'error 'append 'replace) 'error])
                       logger?]
              @defproc[(csv-logger [dir path-string?]
                                   [#:exists exists
                                    (or/c 'error 'append 'replace) 'error])
                       logger?])]{
Write @tt{metrics.jsonl} or @tt{metrics.csv} and
@tt{hparams.rktd} under @racket[dir].}

@section{Callbacks}

@(define-syntax-parse-rule (defproc-lambda-callback body ...)
   #:with (argument ...)
   (for/list ([h (in-list (filter callback-hook? definable-hooks))])
     (define name (hook-name h))
     (define procedure/c
       (if (hook-batch? h)
           'procedure?
           `(procedure-arity-includes/c ,(add1 (length (hook-formals h))))))
     (datum->syntax this-syntax
                    `[,(string->keyword (symbol->string name)) ,name
                      (or/c #f ,procedure/c) #f]))
   (defproc (lambda-callback
             [#:state-key state-key state-key/c "lambda-callback"]
             argument ...)
     callback?
     body ...))

@defproc-lambda-callback{
Lightning's @tt{LambdaCallback}: one keyword per callback hook of
@secref["train-hooks"]. Each procedure takes the run first, then the hook's
arguments.}

@section{Optimization}

@defproc[(scheduler-config [s scheduler?]
                           [#:interval interval interval/c 'epoch]
                           [#:frequency frequency exact-positive-integer? 1])
         scheduler-config?]{
Lightning's @tt{lr_scheduler} dictionary: steps @racket[s] every
@racket[frequency] steps or epochs. Return it from
@racket[define-optimizers].}

@defproc[(scheduler-config? [v any/c]) boolean?]{
Recognizes @racket[scheduler-config] values.}

@deftogether[(@defproc[(hparam [tr trainer?] [name symbol?]) real?]
              @defproc[(set-hparam! [tr trainer?] [name symbol?] [v real?])
                       void?]
              @defproc[(trainer-hparams [tr trainer?])
                       (hash/c symbol? real? #:immutable #t)])]{
Read and write the fields named by @racket[define-hparams]; another
@racket[name] is a contract violation.}

@deftogether[(@defproc[(manual-backward! [r manual-run/c] [loss loss/c])
                       void?]
              @defproc[(manual-step! [r manual-run/c] [opt optimizer?]) void?]
              @defproc[(manual-zero-grads! [r manual-run/c] [opt optimizer?])
                       void?])]{
Manual optimization, inside a @racket[define-training-step] of a trainer
defined with @racket[#:automatic-optimization? #f].
@racket[manual-zero-grads!] runs the @racket[on-before-zero-grad] hooks.}

@section{Contracts}

@deftogether[(@defthing[stage/c contract?]
              @defthing[run-stage/c contract?]
              @defthing[limit/c contract?]
              @defthing[val-check-interval/c contract?]
              @defthing[precision/c contract?]
              @defthing[clip-algorithm/c contract?]
              @defthing[fast-dev-run/c contract?]
              @defthing[interval/c contract?]
              @defthing[reduce/c contract?]
              @defthing[state-key/c contract?])]{
The keyword values: @racket['fit], @racket['test] or @racket['predict]; a
run's stage; a batch count or a fraction of the loader; a batch count or a
fraction of the epoch; @racket['32-true], @racket['16-mixed] or
@racket['bf16-mixed]; @racket['norm] or @racket['value]; a boolean or a
batch count; @racket['step] or @racket['epoch]; @racket['mean],
@racket['sum], @racket['max] or @racket['min]; a non-empty string.}

@deftogether[(@defthing[scalar-tensor/c contract?]
              @defthing[metric-value/c contract?]
              @defthing[loss/c contract?]
              @defthing[logger-spec/c contract?])]{
A one-element tensor; a real or such a tensor; such a tensor of a floating
dtype; @racket[#f], a logger or a list of loggers.}

@deftogether[(@defthing[fitting-run/c contract?]
              @defthing[batch-start-run/c contract?]
              @defthing[step-pending-run/c contract?]
              @defthing[logging-run/c contract?]
              @defthing[manual-run/c contract?])]{
A run inside an active @racket[fit]; inside an @racket[on-train-batch-start]
hook; in a training batch whose optimizer step is pending; in a hook that
may log; inside a manual training step.}

@section[#:tag "train-hooks"]{Hooks}

Callbacks are C and the trainer is T. A hook defined by both runs the
callbacks first, in order, then the trainer, except where the table says
otherwise. Arguments follow the run; @tt{batch…} is the batch values.
@racket[log!] gives where a hook may log and its default: @tt{step} and
@tt{epoch} allow both with that default, @tt{epoch only} allows only per
epoch.

@(define (who h)
   (define sides (hook-sides h))
   (cond
     [(not (memq 'trainer sides)) "C"]
     [(not (memq 'callback sides)) "T"]
     [(eq? (hook-order h) 'trainer-first) "T, C"]
     [(eq? (hook-order h) 'by-kind) "C plain, T, C monitor and checkpoint"]
     [else "C, T"]))
@(define (arguments h)
   (define names (map symbol->string (hook-formals h)))
   (string-join (if (hook-batch? h) (append names (list "batch…")) names)))
@(define (log-cell h)
   (define rule (hook-log h))
   (cond
     [(equal? rule step-log) "step"]
     [(equal? rule epoch-log) "epoch"]
     [(equal? rule epoch-only-log) "epoch only"]
     [else "no"]))
@(define (defined-by h)
   (if (hook-form h)
       (racketidfont (symbol->string (hook-form h)))
       (tt "define-hook")))
@(define (hook-row h)
   (list (tt (symbol->string (hook-name h))) (defined-by h) (who h)
         (arguments h) (log-cell h)))

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (cons (list @bold{hook} @bold{defined by} @bold{who}
                     @bold{arguments} @bold{@racket[log!]})
               (map hook-row hooks))]

The order of one @racket[fit], after Lightning 2.6.5, with
@italic{A} the @racket[#:accumulate-grad-batches]:

@itemlist[
 @item{@tt{configure-callbacks}; @tt{setup}; the device move;
       @tt{configure-optimizers}; @tt{on-fit-start}.}
 @item{With @racket[#:val] and sanity steps: @tt{on-sanity-check-start},
       a validation loop whose metrics are discarded,
       @tt{on-sanity-check-end}.}
 @item{@tt{on-train-start}, then per epoch @tt{on-train-epoch-start} and per
       batch: the transfer hooks, @tt{on-train-batch-start}, the training
       step, @tt{on-before-zero-grad} on a window's first batch,
       @tt{on-before-backward} with the loss ÷ @italic{A},
       @tt{on-after-backward}, @tt{on-before-optimizer-step} at a window's
       end, the schedulers, @tt{on-train-batch-end}, and a validation loop
       when one is due.}
 @item{@tt{on-train-epoch-end} after the epoch's validation loop, then
       @tt{on-train-end}, @tt{on-fit-end} and @tt{teardown}.}
 @item{A validation loop runs in eval mode with gradients off:
       @tt{on-validation-start}, @tt{on-validation-epoch-start}, per batch
       the transfer hooks, @tt{on-validation-batch-start}, the validation
       step and @tt{on-validation-batch-end}, then
       @tt{on-validation-epoch-end} and @tt{on-validation-end}.}]
