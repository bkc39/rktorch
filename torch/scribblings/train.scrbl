#lang scribble/manual

@(require (for-syntax racket/base
                      (only-in syntax/parse/pre id this-syntax)
                      (only-in torch/train/private/hooks
                               callback-hook? definable-hooks hook-batch?
                               hook-formals hook-name))
          (only-in syntax/parse/define define-syntax-parse-rule)
          "common.rkt"
          torch/scribblings/labels
          (for-label (only-in torch/data/loader dataloader dataloader?
                              tensor-dataset)))

@title[#:tag "train"]{Training}

@defmodule[torch/train]

A trainer is a layer whose body forms are a method's steps and hooks, and
@racket[fit] runs it in Lightning 2.6.5's loop order. The checkpoint callbacks
of @secref["train-checkpoints"], @racket[#:ckpt-path] @racket['last] and
@racket['best], and resuming from a mid-epoch checkpoint are not implemented
yet: they raise @racket[exn:fail:unsupported].

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

@examples[#:eval train-eval
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
(real? (run-metric r 'train/loss))]}

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
@racket[run] is not bound. The logger receives each argument that is a real,
string, symbol, boolean or list of those; the trainer keeps no other
argument.}

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
checkpoints carry them. Each must hold a real once @racket[define-init]
returns; otherwise constructing the trainer raises @racket[exn:fail:contract]
naming the field.}

@deftogether[(@defform[(define-training-step step-formals body ...+)]
              @defform[(define-validation-step step-formals body ...+)]
              @defform[(define-test-step step-formals body ...+)]
              @defform[(define-predict-step step-formals body ...+)])]{
The steps; @racket[step-formals] receive the values of one batch. The
training step answers a loss (a floating @racket[scalar-tensor/c]),
@racket[(values loss extra)], or @racket[#f], which skips that batch's
backward pass; the window's optimizer step still runs, on the gradients of
its other batches. The other steps answer one value, which reaches
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
              [#:ckpt-path ckpt-path ckpt-path/c #f])
         run?]{
Trains @racket[trainer] in the order of @secref["train-hooks"] and answers
the finished run. Keywords follow Lightning's @tt{Trainer}. A
@racket[#:device] of @racket[#f] leaves the trainer where it is and sends
each batch to the trainer's device. @racket[log!] reduces on the device and
reads to the host at the end of every @racket[#:log-every-n-steps]-th
accumulation window of the fit, counted across epochs, at the step of a
@racket[request-stop!], and at each epoch end, in one transfer for all keys.
The logger gets a training row, and a validation epoch's row, labelled with
the windows ended before the current one, where under manual optimization
each batch ends one whatever it steps; the first window's row is at step
0, and an epoch's training row before any window ends is at step -1. Each row carries
the key @racket['epoch], @racket[run-epoch], unless one was logged. A
per-step value logged in a validation loop reaches the logger at each
validation batch, apart from the training values and without
@racket['epoch], labelled 0, 1, … with the validation batches the fit has run
outside the sanity check; in @racket[test], with the batches of that call.
@racket[#:max-steps] bounds @racket[run-global-step]; a @racket[skip-epoch!] at
every epoch's first batch ends no window, so a fit bounded by
@racket[#:max-steps] alone does not end, as in Lightning. @racket[#:fast-dev-run] runs that
many batches (one for @racket[#t]) of one epoch, with no sanity check, no loggers and no
@racket[monitor] or @racket[checkpoint] callbacks. If an @racket[exn:fail]
or @racket[exn:break] is raised, @racket[fit] runs each callback's
@racket[on-exception] and then @racket[teardown] with breaks disabled, and
re-raises the exception; a break issued meanwhile is absorbed if breaks were
enabled when @racket[fit] was called, and stays pending otherwise. A loop that
leaves a loader early stops its @racket[#:workers] threads. As Lightning's fetcher, a
loop pulls a sized loader without @racket[#:workers] no further than its last batch, so
a shuffled loader that its batch size divides evenly orders its next epoch otherwise
than a loop run to exhaustion; a loader with @racket[#:workers] prefetches to the end,
as torch's multi-process loader does.
@racket[#:precision] @racket['auto-mixed] picks @racket['bf16-mixed] where the
device supports it and @racket['16-mixed] otherwise; @racket['16-mixed] scales
the loss with a @racket[grad-scaler], unscales before
@racket[on-before-optimizer-step], and on the CPU runs as
@racket['bf16-mixed], as Lightning does. A mixed precision runs the training
and validation steps under @racket[call-with-autocast] and no other hook, with
@racket[#:cache-enabled? #f] as Lightning, so a @racket[manual-step!] reaches
the step's later forwards. Under a scaler, a window whose last batch returns
@racket[#f] runs @racket[on-before-optimizer-step] and the clip on the still
scaled gradients, then neither steps nor updates the scale, and counts as a
step, as Lightning does. @racket[#:gradient-clip-val] clips each optimizer's
parameters after @racket[on-before-optimizer-step], by
@racket[clip-grad-norm!] or, for @racket['value], @racket[clip-grad-value!];
under a scaler, clipping with a fused optimizer raises before the first batch,
where Lightning raises at the clip.

@examples[#:eval train-eval
(define-trainer Mixed (net)
  (define-init () (set! net (Linear 2 1)))
  (define-optimizers (sgd (parameters net) #:lr 0.1))
  (define-training-step (x y) (mse-loss (net x) y)))
(define mixed (fit (Mixed) (dataloader (tensor-dataset (randn 8 2) (randn 8 1)) #:batch-size 4)
                   #:max-epochs 1 #:logger #f #:precision 'auto-mixed
                   #:gradient-clip-val 0.5))
(list (run-precision mixed) (run-grad-scaler mixed))
]

@racket[#:ckpt-path] resumes from a checkpoint; @secref["train-hooks"] says
what is restored at which hook. A checkpoint from an epoch end resumes at the
next epoch and one from the fit's end at the epoch it ended on; the fit then
runs to @racket[#:max-epochs]. The sanity check runs on a resumed fit, after
the callback states are restored, as in Lightning.
@itemlist[
 @item{Refused before any hook runs: a checkpoint taken outside an epoch end
       and the fit's end; one whose loop counters, hyperparameters, logged
       values, logging book, RNG states, grad-scaler state or optimizer and
       scheduler states are missing or malformed; one that resumes at an epoch
       above @racket[#:max-epochs]; one saved under a
       @racket[current-generator] when the one in effect as @racket[fit] is
       called is @racket[#f] or on another device.}
 @item{Refused as it is restored: a trainer that does not hold the
       checkpoint's parameters and buffers, strictly, which leaves its
       parameters and hyperparameters unchanged; a trainer that configures a
       different number of optimizers or schedulers, or an optimizer of another
       kind; a @racket[#:weights-only?] checkpoint, with Lightning's message
       when the optimizers are restored; a device stream's state (the
       @tt{"cuda"} entry) its device refuses, before any other stream is
       restored, since before any hook only its dtype and shape are checked.
       Every refusal names the file and keeps the error's kind.}
 @item{Skipped with a warning on the @racket['rktorch-train] topic and stderr:
       a saved hyperparameter the trainer does not define, a callback
       state no callback claims, and a saved @racket[current-generator] state
       when a hook has set the parameter to @racket[#f] by the time it is
       restored. A save that is not @racket[#:weights-only?],
       from a @racket[#:kind checkpoint] or @racket[#:kind monitor] callback
       in @tt{on-train-epoch-end}, warns in the saving run when that callback
       or a later one logs after it in the same hook.}
 @item{Restored: the default streams, @racket[current-generator]'s and
       Racket's @racket[current-pseudo-random-generator], so a resumed fit
       draws what the uninterrupted one would have, which Lightning does not
       do. The values logged and not yet published are restored
       with what @racket[run-metric] and @racket[run-metrics] read, so the rows
       logged after the resume and the values read are the uninterrupted
       fit's.}
 @item{Not restored: a loader's own @racket[#:generator]; a callback whose
       @tt{state-dict} answers its @racket[generator-state] and whose
       @tt{load-state!} calls @racket[set-generator-state!] carries it.}
 @item{Not in the file: what is logged in @tt{on-train-epoch-end} after the
       save, which the resumed epoch reads as the value logged an epoch
       earlier, and what a later callback's @tt{on-train-epoch-end} changes
       in its state. Lightning 2.6.5 behaves the same. By where the save is
       made:
       @itemlist[
        @item{from a plain callback or the trainer's own hook: the save comes
              before or during the trainer's logging, and the resume warns;}
        @item{from a @racket[#:kind checkpoint] or @racket[#:kind monitor]
              callback, or @racket[model-checkpoint]: the resume is exact
              when the save comes after every callback that logs or changes
              its state in @tt{on-train-epoch-end}. A later log warns in the
              saving run, since the file cannot know; a later state change
              does not warn.}]}]

@examples[#:eval train-eval
(define-trainer Resumed (net)
  (define-init () (set! net (Linear 2 1)))
  (define-optimizers (sgd (parameters net) #:lr 0.1))
  (define-training-step (x y) (mse-loss (net x) y)))
(define resume-path (build-path (find-system-path 'temp-dir) "resumed.ckpt"))
(define resume-data (dataloader (tensor-dataset (randn 8 2) (randn 8 1)) #:batch-size 4))
(define-callback epoch-saver ()
  #:kind checkpoint
  (define-hook (on-train-epoch-end) (save-checkpoint! run resume-path)))
(void (fit (Resumed) resume-data #:max-epochs 2 #:logger #f
           #:callbacks (list (epoch-saver))))
(define resumed (fit (Resumed) resume-data #:max-epochs 3 #:logger #f
                     #:ckpt-path resume-path))
(list (run-epoch resumed) (run-global-step resumed))
(delete-file resume-path)
]

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
               [#:ckpt-path ckpt-path ckpt-path/c #f])
         run?]{
The test loop: @racket[setup] @racket['test], the @racket[on-test-*] hooks,
@racket[teardown]. The trainer needs a @racket[define-test-step].
@racket[#:ckpt-path] restores the parameters, hyperparameters and callback
states only.}

@defproc[(predict [trainer trainer?]
                  [loader dataloader?]
                  [#:device device (or/c #f device?) #f]
                  [#:callbacks callbacks (listof callback?) '()]
                  [#:limit-predict-batches limit limit/c 1.0]
                  [#:ckpt-path ckpt-path ckpt-path/c #f])
         list?]{
The predict loop; answers the predict step's output per batch. Nothing is
logged. The trainer needs a @racket[define-predict-step].
@racket[#:ckpt-path] restores the parameters, hyperparameters and callback
states only.}

@deftogether[(@defproc[(trainer? [v any/c]) boolean?]
              @defproc[(callback? [v any/c]) boolean?])]{
Recognize the instances of @racket[define-trainer] and
@racket[define-callback] types.}

@section[#:tag "train-checkpoints"]{Checkpoints}

@defproc[(save-checkpoint! [r run?] [path path-string?]
                           [#:weights-only? weights-only? boolean? #f])
         void?]{
Lightning's @tt{Trainer.save_checkpoint}. It writes:
@itemlist[
 @item{the trainer's parameters and buffers, its @racket[define-hparams] fields
       and @racket[define-init] arguments;}
 @item{the loop's position and counters, and, for a save in
       @tt{on-train-epoch-end}, whether the trainer's hook had run
       (@tt{"epoch_end_logged"});}
 @item{@racket[run-metrics], and the values logged and not yet published
       (@tt{"metric_book"}), which a resumed fit publishes and reads as the
       uninterrupted one would;}
 @item{each optimizer's and scheduler's state, and the grad scaler's;}
 @item{every callback's @tt{state-dict} under its state key;}
 @item{the RNG states: the CPU stream, the run's device stream,
       @racket[current-generator]'s, Racket's
       @racket[current-pseudo-random-generator]. A loader's own
       @racket[#:generator] is not among them.}]

The file is safetensors with Lightning's key names; @racket[load-state!] reads
the model out of it with @racket[#:rename]. The callbacks' @tt{state-dict}
hooks run first, then each callback's @tt{on-save-checkpoint} and the
trainer's with the whole checkpoint as a mutable hash they may change. The file
is written to a temporary name and renamed, so a failed save leaves
@racket[path] as it was.

@racket[#:weights-only?] keeps the model, the hyperparameters and the loop's
position, as Lightning's @tt{weights_only}. A checkpoint taken at an epoch end,
a batch end, a validation end, an exception or after the fit can be resumed;
one taken elsewhere is written but @racket[fit] refuses it.}

@defproc[(read-checkpoint [path path-string?])
         (hash/c string? checkpoint-value/c #:immutable #t)]{
The checkpoint at @racket[path] as the hash the @tt{on-save-checkpoint} hooks
saw, with every hash immutable and every tensor on the CPU.}

@defthing[checkpoint-value/c flat-contract?]{
The values a checkpoint holds: booleans, exact rationals, flonums (including
infinities, NaN and @racket[-0.0]), strings, interned symbols, tensors, pairs and
lists of these, and @racket[eq?], @racket[eqv?] or @racket[equal?] hashes of these
that hold their keys strongly and whose string keys do not begin with @tt{$}. A
value that contains itself is not one. They read back @racket[equal?], with hashes
immutable and of the same comparison.}

@defproc[(load-from-checkpoint [make-trainer (unconstrained-domain-> trainer?)]
                               [path path-string?]
                               [#:device device (or/c #f device?) #f]
                               [#:strict? strict? boolean? #t]
                               [#:init-args init-args (hash/c symbol? any/c #:immutable #t) (hasheq)])
         trainer?]{
Lightning's @tt{load_from_checkpoint}: applies @racket[make-trainer] to the
@racket[define-init] arguments the checkpoint saved, with @racket[init-args]
replacing them by formal name, sets the @racket[define-hparams] fields, runs the
trainer's @tt{on-load-checkpoint} with a run that has not started, loads the
parameters and buffers, strictly unless @racket[strict?] is @racket[#f], and
moves the trainer to @racket[device]. An argument that was not a
@racket[checkpoint-value/c], such as a layer, was not saved; without an entry in
@racket[init-args] it raises an error naming the formal. A name in
@racket[init-args] that is not a formal is ignored, as Lightning ignores a
keyword its constructor does not take.

@examples[#:eval train-eval
(define-trainer Kept (net rate)
  (define-init (width #:rate [r 0.1]) (set! net (Linear 2 width)) (set! rate r))
  (define-hparams rate)
  (define-optimizers (sgd (parameters net) #:lr rate))
  (define-training-step (x y) (mse-loss (net x) y)))
(define kept-path (build-path (find-system-path 'temp-dir) "kept.ckpt"))
(define (save-at-end r) (save-checkpoint! r kept-path))
(void (fit (Kept 1 #:rate 0.05)
           (dataloader (tensor-dataset (randn 8 2) (randn 8 1)) #:batch-size 4)
           #:max-epochs 2 #:logger #f
           #:callbacks (list (lambda-callback #:on-train-end save-at-end))))
(define tree (read-checkpoint kept-path))
(list (hash-ref tree "global_step") (hash-ref tree "init_formals"))
(trainer-hparams (load-from-checkpoint Kept kept-path))
(delete-file kept-path)
]}

@defproc[(model-checkpoint [#:dirpath dirpath path-string?]
                           [#:filename filename (or/c #f checkpoint-filename/c) #f]
                           [#:monitor monitor (or/c #f symbol?) #f]
                           [#:verbose? verbose? boolean? #f]
                           [#:save-last save-last boolean? #f]
                           [#:save-top-k save-top-k save-top-k/c 1]
                           [#:save-weights-only? weights-only? boolean? #f]
                           [#:mode mode (or/c 'min 'max) 'min]
                           [#:auto-insert-metric-name? auto-insert? boolean? #t]
                           [#:every-n-train-steps every-n-train-steps
                                                  (or/c #f exact-nonnegative-integer?) #f]
                           [#:every-n-epochs every-n-epochs (or/c #f exact-nonnegative-integer?) #f]
                           [#:save-on-train-epoch-end on-epoch-end (or/c 'auto boolean?) 'auto]
                           [#:enable-version-counter? version-counter? boolean? #t])
         model-checkpoint?]{
Lightning's @tt{ModelCheckpoint}, a @racket['checkpoint] callback. It saves with
@racket[save-checkpoint!] into @racket[dirpath] every @racket[#:every-n-epochs]
epochs (1 when neither trigger is given) or every @racket[#:every-n-train-steps]
optimizer steps, at most once per @racket[run-global-step], and keeps the
@racket[#:save-top-k] best by @racket[#:monitor] under @racket[#:mode] (all for
@racket[-1], none for 0; NaN counts as worst). A @racket[#:monitor] key missing
when it saves warns until a validation loop has run and then raises.

An epoch-triggered save happens in @tt{on-train-epoch-end} when validation runs
every epoch at the epoch end or not at all, and otherwise in
@tt{on-validation-end}, unless @racket[#:save-on-train-epoch-end] says which.

Files are named from @racket[#:filename], by default
@tt{epoch=}@italic{E}@tt{-step=}@italic{S}@tt{.ckpt}:
@itemlist[
 @item{@tt["{name}"] inserts a metric, @tt{epoch} or @tt{step}, written
       @tt{name=}value unless @racket[#:auto-insert-metric-name?] is
       @racket[#f];}
 @item{@tt["{name:.2f}"] and @tt["{name:02d}"] format it;}
 @item{a name already taken gets @tt{-v1}, @tt{-v2}, …;}
 @item{@racket[#:save-last] also writes @tt{last.ckpt} at every save.}]

Its state, under Lightning's state key, carries the best and @italic{k}-th paths
and scores. It is restored in full only when @racket[dirpath] is the one it was
saved from, and otherwise only the best path, with a warning.}

@deftogether[(@defproc[(model-checkpoint? [v any/c]) boolean?]
              @defproc[(model-checkpoint-best-model-path [cb model-checkpoint?]) (or/c #f string?)]
              @defproc[(model-checkpoint-best-model-score [cb model-checkpoint?]) (or/c #f flonum?)]
              @defproc[(model-checkpoint-best-k-models [cb model-checkpoint?])
                       (hash/c string? flonum? #:immutable #t)]
              @defproc[(model-checkpoint-last-model-path [cb model-checkpoint?]) (or/c #f string?)])]{
Lightning's @tt{best_model_path}, @tt{best_model_score}, @tt{best_k_models} and
@tt{last_model_path}; @racket[#f] before the first save.}

@defproc[(on-exception-checkpoint [#:dirpath dirpath path-string?]
                                  [#:filename filename (and/c string? (not/c "")) "on_exception"])
         callback?]{
Lightning's @tt{OnExceptionCheckpoint}: in @tt{on-exception} it saves to
@racket[dirpath]/@racket[filename]@tt{.ckpt}, and a fit that finishes deletes
that file in @tt{teardown}. After a native fault has disabled the library it
writes nothing and warns, leaving any earlier file in place.}

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
The loop state. @racket[run-stage] is the entry's stage from @racket[setup]
to @racket[teardown], and @racket[#f] once the run has finished.
@racket[run-epoch] is Lightning's @tt{current_epoch}.
@racket[run-global-step] counts the accumulation windows ended, also a
window whose step @racket[skip-step!] dropped or whose every loss was
@racket[#f].
@racket[run-stepped?] is whether the current training batch ended one.
@racket[run-metric] reads one key to the host: inside an epoch the latest
step value, in an epoch-end hook and after the epoch its reduction, and for a
key logged both per step and per epoch the newer of the two. Inside the
training loop a per-step key answers the training loop's latest value, also
when a validation loop logs the same key, and after a loop's epoch closes its
reduction answers in that loop's hooks over the other loop's values. A key logged
only per epoch answers the running reduction of the current epoch, read once
until the next @racket[log!] of that key.}

@deftogether[(@defproc[(run-schedulers [r run?]) (listof scheduler-config?)]
              @defproc[(run-estimated-stepping-batches [r fit-run/c])
                       exact-nonnegative-integer?])]{
The schedulers @racket[define-optimizers] returned, as Lightning's
@tt{lr_scheduler_configs}, a bare schedule as an @racket['epoch] one, and
Lightning's @tt{estimated_stepping_batches}, which @racket[define-optimizers]
can read. A test or predict run has no schedulers.}

@deftogether[(@defproc[(run-precision [r run?]) (or/c '32-true '16-mixed 'bf16-mixed)]
              @defproc[(run-grad-scaler [r run?]) (or/c #f grad-scaler?)]
              @defproc[(run-accumulate-grad-batches [r run?]) exact-positive-integer?])]{
The precision @racket[fit] resolved @racket[#:precision] to, the scaler a
@racket['16-mixed] run steps through, and the accumulation factor of the
current epoch.}

@defproc[(request-stop! [r fitting-run/c]) void?]{
Lightning's @tt{should_stop}: the current batch finishes, then the
validation loop and the epoch-end and fit-end hooks run. Requested before an
epoch's first batch, the epoch runs no batch.}

@defproc[(skip-step! [r step-pending-run/c]) void?]{
Drops the pending optimizer step. The window's backward passes still run,
@racket[on-before-optimizer-step] is not called unless it is the caller, and
@racket[run-global-step] and the per-step schedulers still advance.}

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
@racket['mean] is weighted by @racket[#:batch-size], by default the leading
dimension of the first tensor in the batch, searched through nested lists, or 1
without one. The value is detached
before it is accumulated, so a logged loss keeps no graph alive, and, as in
Lightning, a tensor that is not @racket['float32] or @racket['float64] is
converted to @racket['float32] first.}

@defthing[gen:training-logger any/c]{
The generic interface of a logger, with the methods below.}

@deftogether[(@defproc[(training-logger? [v any/c]) boolean?]
              @defproc[(logger-log-hyperparams! [logger training-logger?]
                                                [params (hash/c symbol? any/c)])
                       any]
              @defproc[(logger-log-metrics! [logger training-logger?]
                                            [metrics (hash/c symbol? real?)]
                                            [step (integer-in -1 #f)])
                       any]
              @defproc[(logger-finalize! [logger training-logger?]
                                         [status (or/c 'success 'failed)])
                       any])]{
A logger receives the hyperparameters once at fit start, the host metrics at
each read, and @racket['failed] when the entry point raises.}

@defproc[(stdout-logger [#:out out output-port? (current-output-port)])
         training-logger?]{Prints each metrics read as one line.}

@deftogether[(@defproc[(jsonl-logger [dir path-string?]
                                     [#:exists exists
                                      (or/c 'error 'append 'replace) 'error])
                       training-logger?]
              @defproc[(csv-logger [dir path-string?]
                                   [#:exists exists
                                    (or/c 'error 'append 'replace) 'error])
                       training-logger?])]{
Write @tt{metrics.jsonl} or @tt{metrics.csv} and
@tt{hparams.rktd} under @racket[dir]. A @tt{metrics.csv} row is
@tt{step,key,value}, its key quoted as RFC 4180 has it when it holds a comma,
a quote or a newline.}

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
arguments.
@examples[#:eval train-eval
(define report
  (lambda-callback #:on-train-epoch-end
                   (lambda (r)
                     (printf "epoch ~a: ~a\n" (run-epoch r) (run-metric r 'train/loss)))))
(void (fit (Line) (dataloader (tensor-dataset xs (* 2 xs)) #:batch-size 2)
           #:max-epochs 2 #:logger #f #:callbacks (list report)))]}

@section{Optimization}

@defproc[(scheduler-config [s scheduler?]
                           [#:interval interval interval/c 'epoch]
                           [#:frequency frequency exact-positive-integer? 1])
         scheduler-config?]{
Lightning's @tt{lr_scheduler} dictionary. An @racket['epoch] scheduler
steps at the end of every @racket[frequency]-th epoch; a @racket['step]
scheduler steps at the end of an accumulation window whose last batch is a
multiple of @racket[frequency] batches into the epoch, also when the window's
step was dropped, so with @racket[#:accumulate-grad-batches] 2 and
@racket[frequency] 2 it steps at every window end. Return it from @racket[define-optimizers].}

@defproc[(scheduler-config? [v any/c]) boolean?]{
Recognizes @racket[scheduler-config] values.}

@deftogether[(@defproc[(scheduler-config-scheduler [c scheduler-config?]) scheduler?]
              @defproc[(scheduler-config-interval [c scheduler-config?]) interval/c]
              @defproc[(scheduler-config-frequency [c scheduler-config?])
                       exact-positive-integer?])]{
The fields of a @racket[scheduler-config].}

@deftogether[(@defproc[(hparam [tr trainer?] [name symbol?]) real?]
              @defproc[(set-hparam! [tr trainer?] [name symbol?] [v real?])
                       void?]
              @defproc[(trainer-hparams [tr trainer?])
                       (hash/c symbol? real? #:immutable #t)])]{
Read and write the fields named by @racket[define-hparams]; another
@racket[name] is a contract violation. The next hook reads the written value
from the field. Reading a field a hook has set to a non-real raises an error
naming it.
@examples[#:eval train-eval
(define-trainer Tuned (net lr)
  (define-init () (set! net (Linear 1 1)) (set! lr 0.1))
  (define-hparams lr)
  (define-optimizers (sgd (parameters net) #:lr lr))
  (define-training-step (x y) (mse-loss (net x) y)))
(define tuned (Tuned))
(set-hparam! tuned 'lr 0.5)
(list (hparam tuned 'lr) (trainer-hparams tuned))
]}

@deftogether[(@defproc[(manual-backward! [r manual-run/c] [loss loss/c])
                       void?]
              @defproc[(manual-step! [r manual-run/c] [opt optimizer?]) void?]
              @defproc[(manual-zero-grads! [r manual-run/c] [opt optimizer?])
                       void?])]{
Manual optimization, inside a @racket[define-training-step] of a trainer
defined with @racket[#:automatic-optimization? #f]; the loop then zeroes,
backpropagates, clips and steps nothing, and steps no schedule; it still
clears every gradient before a validation within the epoch, as Lightning's
@tt{on_validation_model_zero_grad} does after each manual batch.
@racket[manual-zero-grads!] runs the @racket[on-before-zero-grad] hooks, then
@racket[zero-grads!]. @racket[manual-backward!] scales the loss under a
@racket['16-mixed] scaler and runs @racket[backward!] between
@racket[on-before-backward] and @racket[on-after-backward].
@racket[manual-step!] unscales, runs @racket[on-before-optimizer-step], steps
and advances @racket[run-global-step], once per call. @racket[opt] must be one
of the optimizers @racket[define-optimizers] returned, as Lightning's
@tt{LightningOptimizer} wraps them.
@examples[#:eval train-eval
(define-trainer Halves (net)
  #:automatic-optimization? #f
  (define-init () (set! net (Linear 2 1)))
  (define-optimizers (values (sgd (parameters net) #:lr 0.1) (sgd (parameters net) #:lr 0.01)))
  (define-training-step (x y)
    (for ([opt (in-list (run-optimizers run))])
      (manual-zero-grads! run opt)
      (manual-backward! run (mse-loss (net x) y))
      (manual-step! run opt))
    #f))
(run-global-step (fit (Halves) (dataloader (tensor-dataset (randn 8 2) (randn 8 1))
                                           #:batch-size 4)
                      #:max-epochs 1 #:logger #f))
]}

@defproc[(hparam-scheduler [name symbol?]
                           [curve (-> exact-nonnegative-integer? real?)]
                           [#:interval interval interval/c 'step])
         callback?]{
Sets the trainer's hparam @racket[name] to @racket[(curve (run-global-step run))]
at each training batch start, or to @racket[(curve (run-epoch run))] at each
epoch start. It holds nothing between hooks: the value depends only on the step or
epoch, so one instance serves any number of fits, one at a time.}

@defproc[(param-group-scheduler [key schedulable-key/c]
                                [curve (-> exact-nonnegative-integer? real?)]
                                [#:optimizer optimizer exact-nonnegative-integer? 0]
                                [#:groups groups
                                 (or/c #f (non-empty-listof exact-nonnegative-integer?)) #f]
                                [#:interval interval interval/c 'step])
         callback?]{
The same for @racket[key] of the selected parameter groups (every group for
@racket[#f]) of the @racket[optimizer]-th optimizer, written as an absolute
value. The curve's values are reals for @racket['beta1], @racket['beta2] and
@racket['eps], and @racket[(>=/c 0)] for the other keys; a value the optimizer
rejects, such as a negative RMSprop @racket['eps], raises naming the callback.

@examples[#:eval train-eval
(require (only-in racket/math pi))
(define-trainer Dino (net)
  (define-init () (set! net (Linear 4 2)))
  (define-optimizers (adamw (weight-decay-param-groups net #:weight-decay 0.04) #:lr 1e-3))
  (define-training-step (x y) (mse-loss (net x) y)))
(define wd (lambda (t) (+ 0.04 (* 0.18 (- 1 (cos (* pi (/ t 8))))))))
(define r (fit (Dino) (dataloader (tensor-dataset (randn 16 4) (randn 16 2)) #:batch-size 4)
               #:max-steps 8 #:logger #f
               #:callbacks (list (param-group-scheduler 'weight-decay wd #:groups '(0)))))
(define opt (car (run-optimizers r)))
(list (param-group-ref opt 0 'weight-decay) (param-group-ref opt 1 'weight-decay))
]}

@deftogether[(@defproc[(hparam-scheduler? [v any/c]) boolean?]
              @defproc[(param-group-scheduler? [v any/c]) boolean?])]{
Recognize the schedule callbacks.}

@defproc[(gradient-accumulation-scheduler [scheduling accumulation-scheduling/c])
         callback?]{
Lightning's @tt{GradientAccumulationScheduler}: from each epoch key of
@racket[scheduling] on, accumulate that many batches; a missing epoch 0 means
1. It needs automatic optimization and @racket[#:accumulate-grad-batches] 1,
and raises at @racket[on-train-start] otherwise.
@racket[run-estimated-stepping-batches] still counts with the fit's
@racket[#:accumulate-grad-batches], as Lightning's does.
@examples[#:eval train-eval
(define-trainer Plain (net)
  (define-init () (set! net (Linear 2 1)))
  (define-optimizers (sgd (parameters net) #:lr 0.1))
  (define-training-step (x y) (mse-loss (net x) y)))
(run-global-step (fit (Plain) (dataloader (tensor-dataset (randn 16 2) (randn 16 1))
                                          #:batch-size 2)
                      #:max-epochs 2 #:logger #f
                      #:callbacks (list (gradient-accumulation-scheduler (hasheqv 1 4)))))
]}

@section{Contracts}

@deftogether[(@defthing[stage/c contract?]
              @defthing[run-stage/c contract?]
              @defthing[limit/c contract?]
              @defthing[val-check-interval/c contract?]
              @defthing[precision/c contract?]
              @defthing[clip-algorithm/c contract?]
              @defthing[fast-dev-run/c contract?]
              @defthing[interval/c contract?]
              @defthing[schedulable-key/c contract?]
              @defthing[accumulation-scheduling/c contract?]
              @defthing[reduce/c contract?]
              @defthing[state-key/c contract?]
              @defthing[ckpt-path/c contract?]
              @defthing[save-top-k/c contract?]
              @defthing[checkpoint-filename/c contract?])]{
The keyword values: @racket['fit], @racket['test] or @racket['predict]; a
run's stage; a batch count or a fraction of the loader; a batch count or a
fraction of the epoch; @racket['32-true], @racket['16-mixed],
@racket['bf16-mixed] or @racket['auto-mixed]; @racket['norm] or
@racket['value]; a boolean or a batch count; @racket['step] or
@racket['epoch]; the parameter-group keys a schedule callback can write; an
immutable non-empty hash from epochs to accumulation factors; @racket['mean],
@racket['sum], @racket['max] or @racket['min]; a non-empty string;
@racket[#f], a path, @racket['last] or @racket['best]; an integer of at least
@racket[-1]; a file-name pattern whose fields are @tt["{name}"],
@tt["{name:.Nf}"] or @tt["{name:0Nd}"].}

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
              @defthing[manual-run/c contract?]
              @defthing[fit-run/c contract?])]{
A run inside an active @racket[fit]; inside an @racket[on-train-batch-start]
hook; in a training batch whose optimizer step is pending; in a hook that
may log; inside a manual training step; of @racket[fit], finished or not.}

@include-section["train-hooks.scrbl"]
