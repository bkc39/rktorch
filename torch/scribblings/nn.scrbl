#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch lambda~> prop:to relu tensor? to to-able?)
                     torch/nn
                     torch/private/contract))

@title{Layers}

@defmodule[torch/nn]

@defform[(define-layer name (field ...) clause ... #:forward (input ...) body ...+)
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
                  (code:line #:predicate id)]
          [formal id
                  [id default-expr]
                  (code:line keyword id)
                  (code:line keyword [id default-expr])])
         #:contracts ([contract-expr contract?])]{

Defines a layer: a constructor @racket[name], a predicate @racket[name?],
and a struct with one slot per @racket[field].  An instance is a
@racket[layer?] and applies as a procedure, running @racket[body] with
every field in scope.

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
                  [#:stride pos-size/c #:padding nonneg-size/c]
                  conv2d?)
  #:init (in-channels out-channels kernel-size
          #:stride [stride 1]
          #:padding [padding 0])
  (set! kernel-size (->2d kernel-size))
  (set! stride (->2d stride))
  (set! padding (->2d padding))
  (define shape
    (list out-channels in-channels (car kernel-size) (cadr kernel-size)))
  (set! weight (Parameter (kaiming-uniform shape)))
  (set! bias (Parameter (uniform-init (list out-channels) -0.1 0.1)))
  #:forward (x)
  (conv2d x weight #:bias bias #:stride stride #:padding padding))
]

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
through @racket[to]; a plain tensor field does not.
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
