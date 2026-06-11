#lang scribble/lp2

@(require (for-label (except-in racket/base exp log sqrt max min + - * /)
                     torch))

@section[#:tag "ex-autograd"]{Autograd: gradients by backpropagation}

The calculus hello-world: for @tt{y = sum(x*x)} the gradient is @tt{2x}.
@racket[tensor]'s @racket[#:requires-grad?] marks @tt{x} as a leaf to
differentiate with respect to, @racket[backward!] runs backpropagation from
the scalar @tt{y}, and @racket[grad] reads the accumulated gradient. The
loss pipeline threads left to right: @racket[(~> x (* x) Σ)] is
@racket[(Σ (* x x))], with @racket[Σ] the unicode alias of @racket[sum].

@chunk[<r03-require>
(require torch)]

@chunk[<r03-provide>
(provide run-example)]

@chunk[<r03-run>
(define (run-example)
  (define x (tensor '(1 2 3) #:requires-grad? #t))
  (backward! (~> x (* x) Σ))
  (grad x))]

The matching @filepath{python/03_autograd.py} runs the same computation in
PyTorch; both sides are deterministic, so the cross-test checks exact values.

@chunk[<*>
  <r03-require>
  <r03-provide>
  <r03-run>]
