#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     torch/nn
                     torch/private/contract))

@title{Layers}

@defmodule[torch/nn]

@defform[(define-module name (formal ...) clause ... #:forward (input ...) body ...+)
         #:grammar
         ([formal id
                  [id default-expr]
                  (code:line keyword id)
                  (code:line keyword [id default-expr])]
          [clause (code:line #:coerce ([id expr] ...))
                  (code:line #:params ([id expr] ...))
                  (code:line #:buffers ([id expr] ...))
                  (code:line #:submodules ([id expr] ...))
                  (code:line #:reflection-name expr)
                  (code:line #:contract contract-expr)
                  (code:line #:predicate id)])
         #:contracts ([contract-expr contract?])]{

Defines a layer: a constructor @racket[name] whose arguments are the
@racket[formal]s, and a predicate @racket[name?].  An instance is a
@racket[module?] and applies as a procedure, running @racket[body] with
every constructor argument, parameter, buffer and submodule in scope.

@racket[#:coerce] rebinds a constructor argument before anything else
runs.  @racket[#:params] tensors are marked as requiring gradients and are
what @racket[parameters] returns, own parameters first and then each
submodule's, in declaration order.  @racket[#:buffers] travel with the
layer but are not trained.  @racket[#:submodules] are recursed into by
@racket[parameters], @racket[named-parameters], @racket[buffers],
@racket[train!] and @racket[eval!]; a submodule's parameters are named
under its field, as in @racket["fc1.weight"].

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
(define-module Conv2d (in-channels out-channels kernel-size
                       #:stride [stride 1]
                       #:padding [padding 0])
  #:contract (->* [exact-positive-integer? exact-positive-integer? pos-size/c]
                  [#:stride pos-size/c #:padding nonneg-size/c]
                  conv2d?)
  #:coerce ([kernel-size (->2d kernel-size)]
            [stride (->2d stride)]
            [padding (->2d padding)])
  #:params ([weight (kaiming-uniform (list out-channels in-channels
                                            (car kernel-size) (cadr kernel-size)))]
            [bias (uniform-init (list out-channels) -0.1 0.1)])
  #:forward (x)
  (conv2d x weight #:bias bias #:stride stride #:padding padding))
]

An invariant that relates two arguments is a @racket[->i] precondition
rather than a guard in the body:

@racketblock[
(define-module SelfAttention (n-embd n-head)
  #:contract (->i ([n-embd exact-positive-integer?]
                   [n-head exact-positive-integer?])
                  #:pre (n-embd n-head) (zero? (remainder n-embd n-head))
                  [_ self-attention?])
  ...)
]

Without @racket[#:contract] nothing is exported; a layer local to a model
or a test needs no contract boundary.
}
