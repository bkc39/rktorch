#lang scribble/manual

@(require (only-in racket/list add-between)
          (only-in scribble/racket to-element)
          "common.rkt"
          torch/scribblings/labels)

@(define tiers-eval (make-base-eval))

@(define (exported-names module)
   (module-declared? module #t)
   (define-values (variables syntaxes) (module->exports module))
   (sort (for*/list ([phase+exports (in-list (append variables syntaxes))]
                     [export (in-list (cdr phase+exports))])
           (car export))
         symbol<?))

@(define operator-list
   (add-between (for/list ([name (in-list (exported-names 'torch/operators))])
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
@racketmodfont{threading}. Tensor arithmetic is spelled @racket[add],
@racket[mul] and the other named operators; the rest of
@racketmodname[torch/operators] is a @racket[prefix-in] away.

@examples[#:eval tiers-eval
  (require racket torch/safe (prefix-in t: torch/operators))
  (define x (tensor (list 1.0 4.0 9.0)))
  (mul x 2.0)
  (t:sqrt x)]

@defmodule[torch/operators #:no-declare]

The bindings of @racketmodname[torch] that @racketmodname[racket] also
exports, computed from both modules' exports when the library is built:

@nested[#:style 'inset]{@operator-list}

Each takes a tensor to libtorch and hands anything else to
@racketmodname[racket]'s binding of the same name, so requiring it in place
of @racketmodname[racket]'s changes no numeric result.
