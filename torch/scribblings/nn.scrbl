#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     (only-in torch tensor?)
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
formals, so a stateless layer needs no body.

What a field holds when @racket[init-body] finishes decides what it is:

@itemlist[
 @item{a @racket[Parameter?] is a parameter: returned by
       @racket[parameters], named by @racket[named-parameters], stepped by
       an optimizer and written to the state dict;}
 @item{a @racket[Buffer?] is a buffer: returned by @racket[buffers],
       carried with the layer but not trained;}
 @item{a @racket[layer?] is a child: @racket[parameters],
       @racket[named-parameters], @racket[buffers], @racket[train!] and
       @racket[eval!] recurse into it, and its parameters are named under
       the field, as in @racket["fc1.weight"] (a @racket[LayerList] may
       register under a @racket[#:prefix] instead);}
 @item{@racket[#f] is a declared but absent slot, skipped by all of the
       above, as @tt{register_parameter(name, None)} is;}
 @item{anything else is a plain field, visible to @racket[#:forward] and
       otherwise ignored.}]

@racket[parameters] lists a layer's own parameters first and then each
child's, each group in field declaration order.  @racket[init-body] runs
sequentially, so the order in which parameters draw from the RNG is the
order of the assignments.

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

A container takes its children as a rest argument and holds them in a
@racket[LayerList]:

@racketblock[
(define-layer Sequential (layers)
  #:contract (-> layer? ... sequential?)
  #:init (#:rest ms)
  (set! layers (LayerList ms #:prefix ""))
  #:forward (x)
  (for/fold ([acc x]) ([m (in-list (layer-list->list layers))])
    (m acc)))
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
Recognizes the result of @racket[Buffer].
}

@defproc[(LayerList [layers (listof layer?)]
                    [#:prefix prefix (or/c #f string?) #f])
         layer-list?]{
A layer whose children are @racket[layers], named by index.  Assigned to a
field, it registers under the field name, so its parameters are
@racket["layers.0.weight"] and so on; with @racket[prefix] it registers
under that name instead, and @racket[""] drops the segment altogether, as
@racket[Sequential] does.  Two children of one layer may not register
under the same name; the constructor raises @racket[exn:fail:contract]
when a @racket[prefix] collides with another child.  A layer list is not
applicable; iterate it with @racket[layer-list->list].
}

@defproc[(layer-list? [v any/c]) boolean?]{
Recognizes the result of @racket[LayerList].
}

@defproc[(layer-list->list [ll layer-list?]) (listof layer?)]{
The children of @racket[ll], in order.
}

@defproc[(children [m layer?]) (listof layer?)]{
The direct children of @racket[m], in registration order.  A
@racket[LayerList] counts as one child; its own children are reached
through it.
}

@defproc[(named-children [m layer?]) (listof (cons/c string? layer?))]{
The direct children of @racket[m] with the names they registered under.
}
