#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min)
                     torchrkt))

@section[#:tag "ex-autograd"]{Autograd: gradients by backpropagation}

The calculus hello-world: for @tt{y = sum(x*x)} the gradient is @tt{2x}.
@racket[requires-grad!] marks @tt{x} as a leaf to differentiate with respect
to, @racket[backward!] runs backpropagation from the scalar @tt{y}, and
@racket[grad] reads the accumulated gradient.

@chunk[<r03-require>
(require torchrkt)]

@chunk[<r03-provide>
(provide run-example)]

@chunk[<r03-run>
(define (run-example)
  (define x (requires-grad! (tensor '(1 2 3))))
  (backward! (sum (mul x x)))
  (grad x))]

The matching @filepath{python/03_autograd.py} runs the same computation in
PyTorch; both sides are deterministic, so the cross-test checks exact values.

@chunk[<*>
  <r03-require>
  <r03-provide>
  <r03-run>]
