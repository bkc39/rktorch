#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     torch))

@section[#:tag "ex-matmul"]{Shape manipulation and matrix multiplication}

@racket[arange] enumerates values like @tt{torch.arange} (float32 here), and
@racket[reshape] takes the target shape as ordinary arguments. Together they
build a small matrix with known contents.

@chunk[<r02-require>
(require torch)]

@chunk[<r02-provide>
(provide run-example)]

@bold{The Gram matrix of a 2x3.} @racket[transpose] swaps two dimensions and
@racket[matmul] contracts them, so the result is the 2x2 product
@tt{a @"@" a.T} — again deterministic, for an exact parity check.

@chunk[<r02-run>
(define (run-example)
  (define a (reshape (arange 6) 2 3))
  (matmul a (transpose a 0 1)))]

@chunk[<*>
  <r02-require>
  <r02-provide>
  <r02-run>]
