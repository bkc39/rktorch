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

@racketmodname[torch] extends some of @racketmodname[racket]'s own names to
tensors, so @racket[(require torch)] works under @racketmodname[racket/base]
but not beside @racketmodname[racket] or @racketmodname[racket/list], where
two modules would supply the same name. Two modules split
@racketmodname[torch] along that line; together they are
@racketmodname[torch].

@defmodule[torch/safe #:no-declare]

Every binding of @racketmodname[torch] whose name @racketmodname[racket]
does not export, so it requires beside @racketmodname[racket] and
@racketmodfont{threading}. The tensor operators, @racket[+] and
@racket[sqrt] among them, are a @racket[prefix-in] of
@racketmodname[torch/operators] away.

@examples[#:eval tiers-eval
  (require racket torch/safe (prefix-in t: torch/operators))
  (define x (tensor (list 1.0 4.0 9.0)))
  (t:* x 2.0)
  (t:sqrt x)]

@defmodule[torch/operators #:no-declare]

The bindings of @racketmodname[torch] that @racketmodname[racket] also
exports, computed from both modules' exports when the library is built:

@nested[#:style 'inset]{@operator-list}

Each takes a tensor to libtorch and hands anything else to
@racketmodname[racket]'s binding of the same name, so requiring it in place
of @racketmodname[racket]'s changes no numeric result.
