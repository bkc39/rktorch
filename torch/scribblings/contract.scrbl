#lang scribble/manual

@(require (for-label racket/base
                     racket/contract
                     torch/private/contract))

@title{Contracted definitions}

@defmodule[torch/private/contract]

Two forms that attach a contract to a definition at its definition site,
so the contract sits next to the code it describes rather than in a
@racket[contract-out] block at the exporting facade.  Both are allowed only
at module level, because each expands to a @racket[provide].

@defform*[[(define/contract-out id contract-expr expr)
           (define/contract-out header contract-expr body ...+)]
          #:contracts ([contract-expr contract?])]{

Defines @racket[id] (or the name at the base of @racket[header], as
@racket[define] would) and exports it with @racket[contract-expr] via
@racket[contract-out].  Inside the defining module the binding is
uncontracted, exactly as with @racket[contract-out]: only callers that
import it cross the contract boundary, and blame falls on them.

@racket[contract-expr] may refer to bindings defined later in the module.

@racketblock[
(define/contract-out (safe-div a b)
  (-> number? (not/c zero?) number?)
  (/ a b))
]}

@defform*[[(define/checked-out id contract-expr expr)
           (define/checked-out header contract-expr body ...+)]
          #:contracts ([contract-expr contract?])]{

Like @racket[define/contract-out], but exports the definition plainly and
puts the contracted name in a @racket[checked] submodule:

@racketblock[
(define name expr)
(provide name)
(module+ checked (provide (contract-out [name contract-expr])))
]

A module that requires the defining module reaches the uncontracted
definition.  A module that wants the contract requires
@racket[(submod "the-module.rkt" checked)] instead.  This is the form for
a name the library itself calls: the public facade opts in to the
contract, while the library's own calls to the same name pay nothing.

@racket[module+] accumulates across every use in a file and sees the
enclosing module's bindings, so @racket[contract-expr] may name the
module's imports as freely as in @racket[define/contract-out].

Which form to use is decided by one question: does another module in the
library import this name?  If so, @racket[define/checked-out]; if not,
@racket[define/contract-out].
}
