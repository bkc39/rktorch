#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min)
                     torchrkt))

@section[#:tag "ex-randn"]{Seeding the RNG and sampling a tensor}

The smallest end-to-end thing you can do with the bindings is seed the global
random number generator and draw a tensor of standard-normal samples. This is
exactly the computation our PyTorch parity check mirrors, so it doubles as the
hello-world and the first cross-language correctness guarantee.

@chunk[<r00-require>
(require torchrkt)]

Every example exports a @racket[run-example] thunk so the test harness in
@filepath{examples/test/} and this documentation drive the same code.

@chunk[<r00-provide>
(provide run-example)]

@bold{Seed, then sample.} @racket[manual-seed!] makes the draw reproducible;
@racket[randn] takes the shape as ordinary arguments and returns a
@racket[tensor?] whose native handle is reclaimed by Racket's garbage collector
--- there is nothing to free by hand.

@chunk[<r00-run>
(define (run-example)
  (manual-seed! 0)
  (randn 2 2))]

The companion harness @filepath{test/00-randn.rkt} renders it with
@racket[tensor->string] and reads the values back with @racket[tensor->list];
its @racket[test] submodule checks the shape and element count. The matching
@filepath{python/00_randn.py} samples the same seed/shape in PyTorch, and
@filepath{torchrkt/tests/python-cross-test.rkt} checks the two agree.

@chunk[<*>
  <r00-require>
  <r00-provide>
  <r00-run>]
