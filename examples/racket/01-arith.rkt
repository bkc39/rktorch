#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     torch))

@section[#:tag "ex-arith"]{Building tensors and elementwise arithmetic}

Tensors are usually built from data, not sampled: @racket[tensor] takes a
nested list, infers the shape from the nesting (here 2x2), and copies the
values into a float32 tensor whose handle the garbage collector owns.

@chunk[<r01-require>
(require torch)]

@chunk[<r01-provide>
(provide run-example)]

@bold{Arithmetic reads like math.} @racket[+] and @racket[*] dispatch on
their arguments — plain numbers stay racket/base, tensor operands broadcast
through libtorch (the named @racket[add]/@racket[mul] ops are there too) —
and @racket[relu] is the usual rectifier. The computation below is
@tt{(x + 1) * relu(x)} — deterministic, so the Python parity check is exact
rather than seeded.

@chunk[<r01-run>
(define (run-example)
  (define x (tensor '((1 -2) (3 -4))))
  (* (+ x 1) (relu x)))]

The harness @filepath{test/01-arith.rkt} pins the expected values; the
matching @filepath{python/01_arith.py} computes the same expression in PyTorch
for the cross-test.

@chunk[<*>
  <r01-require>
  <r01-provide>
  <r01-run>]
