#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch backward! lambda~> prop:to relu tensor? to to-able?)
                     (only-in torch/data/loader in-dataloader)
                     torch/nn
                     torch/private/contract))

@title{Layers}

@defmodule[torch/nn]

@defform[(define-layer name (field ...) clause ... #:forward forward-formals body ...+)
         #:grammar
         ([field id
                 [id default-expr]
                 (code:line keyword id)
                 (code:line keyword [id default-expr])]
          [clause (code:line #:init (formal ...) init-body ...)
                  (code:line #:init (formal ... #:rest rest-id) init-body ...)
                  (code:line #:init (formal ... . rest-id) init-body ...)
                  (code:line #:reflection-name expr)
                  (code:line #:contract contract-expr)
                  (code:line #:predicate id)
                  (code:line #:on-move moved-body ...+)]
          [forward-formals (input ...)
                           (input ... . rest-id)]
          [formal id
                  [id default-expr]
                  (code:line keyword id)
                  (code:line keyword [id default-expr])]
          [input id
                 [id : input-contract-expr]])
         #:contracts ([contract-expr contract?]
                      [input-contract-expr contract?])]{

Defines a layer: a constructor @racket[name], a predicate @racket[name?],
and a struct with one slot per @racket[field].  An instance is a
@racket[layer?] and applies as a procedure, running @racket[body] with
every field in scope.  A call with other than one argument per
@racket[input] raises @racket[exn:fail:contract:arity] under
@racket[name], whether made directly, through @racket[forward], or
through @racket[layer-forward].

An @racket[input] written @racket[[id : contract-expr]] states what the
layer accepts there, and a call that does not satisfy it is a contract
violation naming the layer and the contract rather than an error raised
from inside the body: a shape the layer cannot take belongs in the
signature, not in an @racket[unless] guard. The check is built once,
where the layer is defined, so it costs one flat check per call; the
party blamed is the label @tt{caller}, since a layer's forward has no
module boundary of its own to name the caller by. A bare @racket[id]
accepts anything, as before.

@racketblock[
(define-layer BatchNorm2d (weight bias running-mean running-var)
  #:forward ([x : image-batch/c])
  (batch-norm x #:weight weight #:bias bias
              #:running-mean running-mean #:running-var running-var))
]

@racket[#:init] is the constructor body, the analogue of @tt{__init__}.
Its @racket[formal]s are the constructor's arguments, in the grammar of
@racket[define]; the rest argument is spelled @racket[#:rest rest-id] or
as @racket[define]'s dotted tail.  Every field starts as @racket[#f], or as the argument of
the same name when a formal shares it, and @racket[init-body] assigns
fields with @racket[set!].  With @racket[#:init], a field is a bare
identifier.  Without it, the fields are themselves the constructor
formals, so a stateless layer needs no body.  A field name is one
state-dict segment and may not contain a dot.

Every layer starts in the @racket['train] mode, and @racket[train!],
@racket[eval!] and @racket[set-mode!] set a layer's own mode and recurse
into its children.  @racket[body] reads the instance's mode with
@racket[with-mode].  The mode is not a field: it is neither a parameter
nor a buffer, and it is not written to the state dict.  Other
per-instance state that is not a tensor belongs in a plain field holding
a @racket[box]; state that is a tensor belongs in a @racket[Buffer]
updated in place.

@racketblock[
(define-layer Dropout (p)
  #:contract (->* [] [#:p (and/c (>=/c 0) (</c 1))] dropout?)
  #:init (#:p [p 0.5])
  #:forward (x)
  (with-mode (dropout x p (training? mode))))
]

What a field holds when @racket[init-body] finishes decides what it is:

@itemlist[
 @item{a @racket[Parameter?] is a parameter: returned by
       @racket[parameters], named by @racket[named-parameters], stepped by
       an optimizer and written to the state dict;}
 @item{a @racket[Buffer?] is a buffer: returned by @racket[buffers],
       named by @racket[named-buffers], written to the state dict after
       the parameters, but not trained;}
 @item{a @racket[layer?] is a child: @racket[parameters],
       @racket[named-parameters], @racket[buffers], @racket[train!] and
       @racket[eval!] recurse into it, and its parameters are named under
       the field, as in @racket["fc1.weight"];}
 @item{a @racket[Children?] value, from @racket[children-by-index] or
       @racket[children-by-key], splices its entries in as children under
       their own names, and the field name is dropped, as
       @tt{add_module} in a loop would;}
 @item{a @racket[Parameters?] value, from @racket[parameters-by-key],
       splices its entries in as parameters under their own names, for a
       set whose size or naming is decided at construction, as
       @tt{register_parameter} in a loop would;}
 @item{@racket[#f] is a declared but absent slot, skipped by all of the
       above, as @tt{register_parameter(name, None)} is;}
 @item{anything else is a plain field, visible to @racket[#:forward] and
       otherwise ignored.}]

@racket[parameters] lists a layer's own parameters first and then each
child's, each group in field declaration order.  A parameter or child
reachable by more than one path, as when two fields hold the same layer,
is listed once, under the first path, so an optimizer steps it once;
the state dict keeps every path, as PyTorch's does, so a tied model
loads into an untied one.
@racket[init-body] runs sequentially, so the order in which parameters
draw from the RNG is the order of the assignments.

@racket[#:contract] exports the layer.  It provides @racket[name] under
@racket[contract-expr] and the predicate under a lowercase name, both via
@racket[contract-out], so a violation blames the calling module.  The
predicate's export name inserts a hyphen before each uppercase letter that
follows a lowercase letter or a digit, then downcases:
@racket[Linear] exports @racket[linear?], @racket[Conv2d] exports
@racket[conv2d?], @racket[MaxPool2d] exports @racket[max-pool2d?].
@racket[#:predicate] names the exported predicate instead.  Both names
are bound in the defining module, so @racket[contract-expr] may use the
lowercase one as its range.  Like @racket[define/contract-out], the clause
is allowed only at module level.

@racketblock[
(define-layer Conv2d (kernel-size stride padding weight bias)
  #:contract (->* [exact-positive-integer? exact-positive-integer? pos-size/c]
                  [#:stride pos-size/c #:padding nonneg-size/c
                   #:bias? boolean?]
                  conv2d?)
  #:init (in-channels out-channels kernel-size
          #:stride [stride 1]
          #:padding [padding 0]
          #:bias? [bias? #t])
  (set! kernel-size (->2d kernel-size))
  (set! stride (->2d stride))
  (set! padding (->2d padding))
  (define shape
    (list out-channels in-channels (car kernel-size) (cadr kernel-size)))
  (set! weight (Parameter (kaiming-uniform shape)))
  (set! bias
        (and bias? (Parameter (uniform-init (list out-channels) -0.1 0.1))))
  #:forward (x)
  (conv2d x weight #:bias bias #:stride stride #:padding padding))
]

@racket[#:bias?] is @racket[#f] where a batch norm follows, as in
@racket[ResNet]: the normalization's shift subsumes the bias, so the
field holds @racket[#f] and no @tt{bias} entry reaches the state dict.

A container is a layer whose children arrive as a named collection
rather than one per field.  It builds them with
@racket[children-by-index] or @racket[children-by-key] and assigns the
result to a field; the entries register directly on the container, so
@racket[Sequential]'s parameters are @racket["0.weight"] and so on.  A
step may be a plain procedure such as @racket[relu], so a model can mix
layers with the functional interface:

@racketblock[
(define-layer Sequential (steps)
  #:contract (->* [] #:rest (or/c (list/c (listof step/c)) (listof step/c))
                  sequential?)
  #:init (#:rest ms)
  (set! steps (children-by-index (if (and (pair? ms) (list? (car ms)))
                                     (car ms)
                                     ms)))
  #:forward (x)
  (for/fold ([acc x]) ([m (in-layers steps)])
    (layer-forward m acc)))
]

An invariant that relates two arguments is a @racket[->i] precondition
rather than a guard in the body:

@racketblock[
(define-layer SelfAttention (n-embd n-head wq wk wv wo)
  #:contract (->i ([n-embd exact-positive-integer?]
                   [n-head exact-positive-integer?])
                  #:pre (n-embd n-head) (zero? (remainder n-embd n-head))
                  [_ self-attention?])
  #:init (n-embd n-head)
  ...)
]

@racket[#:on-move] runs after @racket[to] has moved the layer, with every
field in scope, and only when the move actually rebound something: the
device and dtype of every parameter and buffer are read either side of it
and compared.  @racket[to] is the identity when nothing changes, so a loop
that defensively moves a model to the device it is already on runs the body
not at all; a device round trip runs it twice, once per move, which reading
the placement after the fact could not detect.  It is for state derived
from where the tensors live --- a cached layout, a handle onto their
storage --- which a move invalidates:

@racketblock[
(define-layer LSTM (spec entries params)
  #:init (input-size hidden-size)
  (code:comment "...")
  #:on-move (forget-flattening! entries)
  #:forward (x . state)
  (with-mode (run spec entries lstm-input x state mode)))
]

Without @racket[#:contract] nothing is exported; a layer local to a model
or a test needs no contract boundary.
}

@defform*[[(with-mode body ...+)
           (with-mode id body ...+)]]{
Allowed only inside a @racket[define-layer] @racket[#:forward] body.
Binds @racket[id], or @racket[mode] when no identifier is given, to the
instance's current mode, a @racket[mode/c] value, and evaluates
@racket[body].  The first form is the identifier-then-body one, so
@racket[(with-mode x)] evaluates @racket[x] with @racket[mode] bound.
The mode is read at each call, so a layer that flips between calls sees
the change.
}

@defthing[mode/c contract?]{
A layer's mode: @racket['train] or @racket['eval].
}

@defproc[(training? [mode mode/c]) boolean?]{
Whether @racket[mode] is @racket['train].
}

@defproc[(evaluating? [mode mode/c]) boolean?]{
Whether @racket[mode] is @racket['eval].
}

@deftogether[(@defproc[(BatchNorm2d [num-features exact-positive-integer?]
                                    [#:eps eps real? 1e-5]
                                    [#:momentum momentum real? 0.1])
                       batch-norm2d?]
              @defproc[(BatchNorm1d [num-features exact-positive-integer?]
                                    [#:eps eps real? 1e-5]
                                    [#:momentum momentum real? 0.1])
                       batch-norm1d?])]{
@tt{nn.BatchNorm2d} and @tt{nn.BatchNorm1d}: normalize each of
@racket[num-features] channels over the batch, scale and shift by a
learned @tt{weight} and @tt{bias}, and keep a @racket[Buffer] running
mean and variance that @racket[step!] does not touch --- the forward
updates them, in @racket['train] mode only, and @racket[eval!] switches
the normalization onto them.  @racket[BatchNorm2d] takes an
@tt{[N C H W]} batch and @racket[BatchNorm1d] takes @tt{[N C]} or
@tt{[N C L]}; another rank is a contract violation naming the layer.
The @tt{num-batches-tracked} buffer counts the batches normalized, in
int64 as torch does.
}

@defproc[(Parameter [t tensor?]) Parameter?]{
Returns @racket[t] as a parameter: the same storage under a tensor subtype
that @racket[define-layer] registers, detached from any autograd graph
that produced @racket[t] and with @racket[requires-grad!] set, so a
parameter is always a leaf that @racket[backward!] populates.
}

@defproc[(Parameter? [v any/c]) boolean?]{
Recognizes the result of @racket[Parameter].  Every parameter is a
@racket[tensor?]; a plain tensor is not a parameter, however it was made.
}

@defproc[(Buffer [t tensor?]) Buffer?]{
Returns @racket[t] as a buffer: the same storage under a tensor subtype
that @racket[define-layer] registers among @racket[buffers], detached
from any autograd graph that produced @racket[t].
}

@defproc[(Buffer? [v any/c]) boolean?]{
Recognizes the result of @racket[Buffer].  A buffer follows its layer
through @racket[to]: it always changes device, and a floating-point buffer
also takes a dtype target; a plain tensor field does neither.
}

@defproc[(procedure->Layer [proc procedure?]
                          [#:parameters params (listof (cons/c child-name/c Parameter?)) '()]
                          [#:buffers bufs (listof (cons/c child-name/c Buffer?)) '()]
                          [#:children kids (listof (cons/c child-name/c layer?)) '()])
         layer?]{
Wraps @racket[proc] as a callable layer. Calls through the layer itself,
@racket[forward], or @racket[layer-forward] pass positional arguments to
@racket[proc] and preserve its return values and exceptions. Keyword
arguments to the wrapped procedure are not supported.

The optional association lists register captured parameters, buffers, and
child layers. Names must be unique across all three lists and must not
contain dots. Parameters precede children's parameters in traversal order;
shared values are deduplicated by identity. Training and evaluation recurse
through registered children, as for @racket[define-layer].

Captures are not discovered automatically. In contrast to OCaml's
@tt{Layer.of_fn}, whose parameters belong to a separate variable store,
this wrapper owns its registered tree. Rebinding a captured variable does
not change that registration. The wrapper has a training mode of its own
that @racket[train!] and @racket[eval!] set, but @racket[proc] cannot read
it, so a mode-dependent step belongs in a registered child such as
@racket[Dropout].

@racketblock[
(define projection (Linear 32 32))
(define drop (Dropout #:p 0.1))
(define block
  (procedure->Layer
   (lambda~> projection relu drop)
   #:children (list (cons "projection" projection)
                    (cons "drop" drop))))
(eval! block)
]

For a stateless operation, use @racket[(procedure->Layer relu)].
}

@defproc[(children-by-index [layers (listof step/c)])
         Children?]{
Names @racket[layers] by position, @racket["0"], @racket["1"] and so on,
for a field of a @racket[define-layer] to splice in as children.  An
element that is a procedure but not a @racket[layer?] becomes a child
with no parameters that applies the procedure to its inputs, so it
keeps its index and appears in @racket[children] like any other.  A
tensor such a procedure closes over is neither a parameter nor a
buffer: nothing trains it or saves it, and it lives as long as the
model does.  A value meant to train belongs in a @racket[Parameter]
field of a @racket[define-layer], or in an explicit registration on
@racket[procedure->Layer].
}

@deftogether[(@defproc[(parameters-by-key
                        [entries (listof (cons/c child-name/c Parameter?))])
                       Parameters?]
              @defproc[(Parameters? [v any/c]) boolean?])]{
The parameter counterpart of @racket[children-by-key]: a field holding
one registers every entry as a parameter under the name paired with it,
and the field's own name is dropped.  For a layer whose parameters are
decided at construction rather than declared one per field --- a
recurrent stack naming its weights @tt{weight_ih_l0} through
@tt{bias_hh_l1_reverse} by its depth and direction --- this is what
@tt{register_parameter} in a loop does.  The same duplicate-name check
applies as to children.
}

@defproc[(children-by-key [entries (listof (cons/c child-name/c step/c))])
         Children?]{
Like @racket[children-by-index], with each child under the name paired
with it.  A layer's constructor raises if two of its children would
register under one name, or if two of its parameters or buffers,
however nested, would flatten to the same state-dict name.
}

@defthing[step/c contract?]{
What a container accepts as a step: @racket[(or/c layer? procedure?)].
}

@defthing[child-name/c contract?]{
A name a child registers under: one non-empty segment without a dot,
as with @tt{add_module}.
}

@defproc[(Children? [v any/c]) boolean?]{
Recognizes the result of @racket[children-by-index] and
@racket[children-by-key].
}

@defproc[(LayerList [layers (listof step/c)]) layer-list?]{
A layer whose children are @racket[layers], named by index and nothing
else.  Assigned to a field, it registers under the field name, so its
parameters are @racket["layers.0.weight"] and so on.  A layer list is
not applicable; iterate it with @racket[in-layers].
}

@defproc[(layer-list? [v any/c]) boolean?]{
Recognizes the result of @racket[LayerList].
}

@defproc[(LayerHash [entries (listof (cons/c child-name/c step/c))])
         layer-hash?]{
A layer whose children are @racket[entries], each under its name, in the
order given.  Assigned to a field @racket[parts], a child @racket["enc"]
has parameters @racket["parts.enc.weight"] and so on.  A layer hash is
not applicable; reach a child with @racket[child-ref] or iterate with
@racket[in-layers].
}

@defproc[(layer-hash? [v any/c]) boolean?]{
Recognizes the result of @racket[LayerHash].
}

@defproc*[([(Sequential [step step/c] ...) sequential?]
           [(Sequential [steps (listof step/c)]) sequential?])]{
A layer that applies each step to the previous step's result.  The
steps are its children, named by index, so its parameters are
@racket["0.weight"] and so on.  The steps are given either as arguments
or as one list, so a model may build them with @racket[for/list].
}

@defproc[(sequential? [v any/c]) boolean?]{
Recognizes the result of @racket[Sequential].
}

@defproc[(in-layers [v (or/c Children? layer?)]) sequence?]{
A sequence of the children of @racket[v], in order, for use in
@racket[for] forms: the entries of a @racket[Children?] value, or the
registered children of a layer.  Unlike @racket[children], a layer
listed more than once is yielded each time, so a tied block in a
@racket[Sequential] applies as many times as it is listed.
}

@defproc[(child-ref [m layer?] [name string?]) (or/c layer? #f)]{
The direct child of @racket[m] registered as @racket[name], or
@racket[#f].
}

@defproc[(children [m layer?]) (listof layer?)]{
The direct children of @racket[m], in registration order, each listed
once however many fields hold it.  A @racket[LayerList] counts as one
child; its own children are reached through it.
}

@defproc[(named-children [m layer?]) (listof (cons/c string? layer?))]{
The direct children of @racket[m] with the names they registered under.
}

@defproc[(set-mode! [m layer?] [mode mode/c]) layer?]{
Sets the mode of @racket[m] and of every layer reachable through its
children, and returns @racket[m].
}

Every layer satisfies @racket[to-able?]: @racket[gen:layer] derives
@racket[prop:to], so @racket[(to m 'cuda)] moves each of
@racket[(parameters m)] and @racket[(buffers m)] in place, keeps every
parameter object and its gradient, and returns @racket[m].  A hand-written
@racket[gen:layer] implementation gets this for whatever its
@racket[layer-parameters] and @racket[layer-buffers] report.  See
@racket[to] for the rules on optimizer state and plain fields.

@defproc[(train! [m layer?]) layer?]{
@racket[(set-mode! m 'train)].
}

@defproc[(eval! [m layer?]) layer?]{
@racket[(set-mode! m 'eval)].
}

@defproc[(layer-mode [m layer?]) mode/c]{
The mode of @racket[m] itself.  A layer's mode is its own:
@racket[eval!] on a child does not change its parent's answer.  A
hand-written @racket[gen:layer] implementation that defines no mode
methods is stateless and reports @racket['train].
}

@defproc[(layer-training? [m layer?]) boolean?]{
@racket[(training? (layer-mode m))].
}

@defproc[(layer-set-mode! [m layer?] [mode mode/c]) void?]{
The @racket[gen:layer] method behind @racket[set-mode!]: sets
@racket[m]'s own mode and recurses into its children.  A hand-written
layer that keeps a mode of its own defines this method and
@racket[layer-mode]; @racket[call-with-mode] restores such a layer
through them.  A layer that defines neither is stateless.
}

@defproc[(call-with-mode [m layer?] [mode mode/c] [thunk (-> any)]) any]{
Records the mode of every layer reachable from @racket[m], sets them all
to @racket[mode], calls @racket[thunk], and restores each layer's own
recorded mode, whether @racket[thunk] returns or raises.  A tree whose
layers were in mixed modes comes back exactly as it was.
}

@defproc[(call-with-eval-mode [m layer?] [thunk (-> any)]) any]{
@racket[(call-with-mode m 'eval thunk)].
}

@defform[(in-mode m mode body ...+)]{
@racket[(call-with-mode m mode (lambda () body ...))].
}

@defform[(in-eval-mode m body ...+)]{
@racket[(call-with-mode m 'eval (lambda () body ...))].
}

@defproc[(ema [model layer?] [average layer?]
              [#:decay decay (real-in 0 1) 0.9999])
         ema?]{
An exponential moving average of @racket[model]'s parameters, kept in
@racket[average]: a second layer of the same architecture, built fresh by
the caller, whose parameters are overwritten with the model's on
construction. Passing the model itself, a layer sharing a parameter
object with it, or one on another device or dtype is a contract
violation; a model moved with @racket[to] needs its average moved the
same way. Two parameters over one storage, which the contract cannot
see, average to themselves: the copy and the update read both tensors
before writing either. Buffers are neither copied nor
averaged, as PyTorch's @tt{AveragedModel} leaves them by default; the
average keeps the buffers it was built with. A diffusion model sampled from its averaged weights rather
than its latest ones gives markedly cleaner images, the reason DDPM
training keeps one; the default decay is that paper's.
}

@defproc[(ema-update! [e ema?]) void?]{
Moves every averaged parameter to @racket[decay] times itself plus
@racket[(- 1 decay)] times the model's, under @racket[with-no-grad]. The
first update copies instead of averaging, as PyTorch's
@tt{AveragedModel} does, so an average built before training tracks the
trained weights rather than the initial draw.
}

@defproc[(ema-average [e ema?]) layer?]{
The averaged layer, the one to evaluate with.
}

@defproc[(ema-decay [e ema?]) (real-in 0 1)]{
The decay @racket[e] was built with.
}

@defproc[(ema? [v any/c]) boolean?]{
Whether @racket[v] is an average built by @racket[ema].
}

@section{Optimizers and schedules}

An optimizer holds a list of parameters and answers to @racket[step!] and
@racket[zero-grads!]; every one keeps its state in place on the parameter's
device and dtype, as the #138 Adam does, and follows a parameter moved with
@racket[to]. The learning rate is the one setting that varies during
training, so every optimizer exposes it through @racket[learning-rate] and
@racket[set-learning-rate!], which is what a schedule writes.

@defproc[(sgd [params (listof tensor?)]
              [#:lr lr (>=/c 0)]
              [#:momentum momentum (>=/c 0) 0]
              [#:nesterov? nesterov? boolean? #f]
              [#:weight-decay weight-decay (>=/c 0) 0])
         sgd?]{
@tt{torch.optim.SGD}: with @racket[momentum] the update blends into a
buffer, copied from the first gradient and thereafter @racket[momentum]
times itself plus the gradient; with @racket[nesterov?] the update looks
one blend ahead, which requires a momentum; @racket[weight-decay] adds that
multiple of the parameter to the gradient before either, torch's L2 form.
}

@defproc[(adam [params (listof tensor?)]
               [#:lr lr (>=/c 0) 1e-3]
               [#:beta1 beta1 real? 0.9]
               [#:beta2 beta2 real? 0.999]
               [#:eps eps real? 1e-8]
               [#:weight-decay weight-decay (>=/c 0) 0])
         adam?]{
@tt{torch.optim.Adam} with bias correction; @racket[weight-decay] is the
L2 form applied to the gradient, as there, not AdamW's decoupled one.
}

@defproc[(rmsprop [params (listof tensor?)]
                  [#:lr lr (>=/c 0) 1e-2]
                  [#:alpha alpha (>=/c 0) 0.99]
                  [#:eps eps (>=/c 0) 1e-8]
                  [#:weight-decay weight-decay (>=/c 0) 0]
                  [#:momentum momentum (>=/c 0) 0])
         rmsprop?]{
@tt{torch.optim.RMSprop}, uncentered: a running average of the squared
gradient decayed by @racket[alpha], the parameter moved by the gradient over
that average's root plus @racket[eps]; with @racket[momentum] the move
accumulates into a buffer that starts at zero.
}

@deftogether[(@defproc[(sgd? [v any/c]) boolean?]
              @defproc[(adam? [v any/c]) boolean?]
              @defproc[(rmsprop? [v any/c]) boolean?]
              @defproc[(optimizer? [v any/c]) boolean?])]{
The optimizer predicates; @racket[optimizer?] holds of every optimizer and
of every schedule.
}

@defproc[(step! [opt optimizer?]) void?]{
Applies one update to every parameter that has a gradient, under
@racket[with-no-grad]; on a schedule, advances it and writes its rate.
}

@defproc[(zero-grads! [opt optimizer?]) void?]{
Zeroes the gradient of every parameter, @tt{optimizer.zero_grad()}.
}

@deftogether[(@defproc[(learning-rate [opt optimizer?]) real?]
              @defproc[(set-learning-rate! [opt optimizer?] [lr (>=/c 0)]) void?])]{
The learning rate the next @racket[step!] will use; on a schedule, its
optimizer's.
}

A schedule wraps an optimizer and answers to @racket[step!] like one: the
rate for step 0 is written at construction, and each @racket[step!] on the
schedule advances its count and writes the rate for it, so a training loop
steps the optimizer and then the schedule as in PyTorch. The rates are the
closed forms of @tt{torch.optim.lr_scheduler}'s constructors of the same
names, pinned against them step for step.

@racketblock[
(define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9 #:weight-decay 5e-4))
(define schedule (one-cycle-lr opt #:max-lr 0.1 #:total-steps (* epochs batches)))
(for* ([epoch (in-range epochs)] [(xb yb) (in-dataloader loader)])
  (zero-grads! opt)
  (backward! (cross-entropy (net xb) yb))
  (step! opt)
  (step! schedule))
]

@defproc[(step-lr [opt optimizer?]
                  [#:step-size step-size exact-positive-integer?]
                  [#:gamma gamma real? 0.1])
         scheduler?]{
The base rate times @racket[gamma] to the power of the number of whole
@racket[step-size] periods elapsed.
}

@defproc[(multi-step-lr [opt optimizer?]
                        [#:milestones milestones (listof exact-nonnegative-integer?)]
                        [#:gamma gamma real? 0.1])
         scheduler?]{
The base rate times @racket[gamma] once per milestone reached.
}

@defproc[(exponential-lr [opt optimizer?] [#:gamma gamma real?]) scheduler?]{
The base rate times @racket[gamma] to the power of the step.
}

@defproc[(cosine-annealing-lr [opt optimizer?]
                              [#:t-max t-max exact-positive-integer?]
                              [#:eta-min eta-min real? 0])
         scheduler?]{
Half a cosine from the base rate at step 0 to @racket[eta-min] at
@racket[t-max], and back up beyond it.
}

@defproc[(linear-lr [opt optimizer?]
                    [#:start-factor start-factor (and/c (>/c 0) (<=/c 1)) 1/3]
                    [#:end-factor end-factor (real-in 0 1) 1]
                    [#:total-iters total-iters exact-positive-integer? 5])
         scheduler?]{
The base rate scaled from @racket[start-factor] to @racket[end-factor]
linearly over @racket[total-iters] steps, and held there: the warmup.
}

@defproc[(one-cycle-lr [opt optimizer?]
                       [#:max-lr max-lr (>/c 0)]
                       [#:total-steps total-steps exact-positive-integer?]
                       [#:pct-start pct-start (and/c (>=/c 0) (</c 1)) 0.3]
                       [#:div-factor div-factor (>/c 0) 25]
                       [#:final-div-factor final-div-factor (>/c 0) 1e4])
         scheduler?]{
@tt{OneCycleLR} with cosine annealing: from @racket[max-lr] over
@racket[div-factor] up to @racket[max-lr] over the first @racket[pct-start]
of @racket[total-steps], then down to the initial rate over
@racket[final-div-factor]. Stepping past @racket[total-steps] is an error,
as there. The base rate of @racket[opt] is not used. Momentum is left
alone, where PyTorch's default cycles it; pass @tt{cycle_momentum=False}
to reproduce this schedule there. A @racket[pct-start] of 1 would put the
peak at the last step and leave the descent no steps to spread over, so
the contract excludes it.
}

@defproc[(lambda-lr [opt optimizer?] [factor (-> exact-nonnegative-integer? real?)])
         scheduler?]{
The base rate times @racket[(factor step)].
}

@deftogether[(@defproc[(scheduler? [v any/c]) boolean?]
              @defproc[(scheduler-step-count [s scheduler?]) exact-nonnegative-integer?]
              @defproc[(scheduler-rate [s scheduler?]) real?]
              @defproc[(scheduler-optimizer-of [s scheduler?]) optimizer?])]{
A schedule, the number of times it has been stepped, the rate it last
wrote to its optimizer, and that optimizer. @racket[scheduler-rate] is
@tt{get_last_lr()}: it reports the rate written at construction or by the
last @racket[step!], and does not call a @racket[lambda-lr] factor again.
}

@section{Checkpoints}

@defproc[(state-dict [model layer?]) (listof (cons/c string? tensor?))]{
The model's parameters and then its buffers, each under its dotted path, in
the order of @tt{nn.Module.state_dict}.  Every registered path is kept, so a
tensor shared by two fields appears under both names.
}

@defproc[(save-state! [model layer?] [path path-string?]) void?]{
Writes @racket[(state-dict model)] to @racket[path] in the safetensors
layout: an 8-byte little-endian header length, a JSON header giving each
entry's @tt{dtype}, @tt{shape} and @tt{data_offsets}, then the tensors'
bytes, little-endian.  Entries are typed @tt{F32}, @tt{F64}, @tt{F16},
@tt{BF16}, @tt{I64}, @tt{U8} or @tt{BOOL}, and every value is written
exactly, a @racket['float64] tensor included.  A tensor of any other dtype is refused
with the name of its entry.
}

@defproc[(load-state! [model layer?]
                      [path path-string?]
                      [#:strict? strict? boolean? #t])
         any]{
Copies the entries of the checkpoint at @racket[path] into
@racket[model]'s parameters and buffers, in place and outside the autograd
tape, as @tt{load_state_dict} does.  A loaded value takes the dtype and
device of the tensor it lands in.

The file is checked against the model before anything is copied.  With
@racket[strict?], a key the model has and the file lacks, or the file has
and the model lacks, is an error, and the one error names every such key;
a strict load returns @|void-const|.  Without it, the entries the two
share are loaded and the result is two values: the missing keys in the
model's order and the unexpected keys in alphabetical order.  An entry
whose shape differs from the model's is an error in either mode, reported
per key with both shapes, because equal element counts do not make shapes
equal.
}
