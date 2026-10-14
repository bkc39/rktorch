#lang scribble/manual

@(require (only-in racket/list add-between)
          (only-in scribble/racket to-element)
          (only-in torch/private/exports export-names)
          "common.rkt"
          torch/scribblings/labels)

@(define tiers-eval (make-base-eval))

@(define operator-list
   (add-between (for/list ([name (in-list (export-names 'torch/operators))])
                  (to-element (datum->syntax (quote-syntax here) name)))
                " "))

@title[#:tag "imports"]{Importing beside racket}

In a module written in @racketmodname[racket/base] or @racketmodname[racket],
@racket[(require torch)] shadows the language's bindings. @racket[+],
@racket[*], @racket[sqrt], @racket[take] and the rest of
@racketmodname[torch/operators] take a tensor to libtorch and hand anything
else to @racketmodname[racket]'s binding.

@examples[#:eval tiers-eval
  (require torch)
  (define x (tensor (list 1.0 4.0 9.0)))
  (* x 2.0)
  (sqrt x)
  (sqrt 4)
  (take (list 1 2 3) 2)]

A require cannot shadow another require, so @racketmodname[torch] does not
require beside @racketmodname[racket] or @racketmodname[racket/list], for
label or otherwise:

@examples[#:eval tiers-eval #:label #f
  (eval:error (module twice racket/base (require racket torch)))]

Two modules split @racketmodname[torch] along that line; together they are
@racketmodname[torch].

@defmodule[torch/safe #:no-declare]

Every binding of @racketmodname[torch] whose name @racketmodname[racket]
does not export, so it requires beside @racketmodname[racket] and
@racketmodfont{threading}, for label too.
@racket[(prefix-in t: torch/operators)] adds the operators beside it.

@defmodule[torch/operators #:no-declare]

The bindings of @racketmodname[torch] that @racketmodname[racket] also
exports, computed from both modules' exports when the library is built:

@nested[#:style 'inset]{@operator-list}

Each takes a tensor to libtorch and hands anything else to
@racketmodname[racket]'s binding of the same name, so requiring it in place
of @racketmodname[racket]'s changes no numeric result.
