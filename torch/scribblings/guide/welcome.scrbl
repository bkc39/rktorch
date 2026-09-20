#lang scribble/manual
@(require "../common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     torch
                     torch/nn))

@title[#:tag "welcome"]{Welcome to rktorch}

If you know PyTorch, most of what you know transfers: the dtypes, the
broadcasting rules, the autograd engine and the kernels are the same ones.
This chapter covers what is different --- how the library is spelled in
Racket --- before the rest of the guide gets to work.

@section[#:tag "welcome-first"]{A first tensor}

@racket[tensor] builds one from ordinary Racket data. A list of lists is a
matrix, and the printed form is PyTorch's:

@torch-examples[
(require torch)
(tensor '((1 2) (3 4)))
]

The library infers the element type from the data, the way PyTorch does ---
integers give an integer tensor, decimals give a single-precision float one:

@torch-examples[
(tensor-dtype (tensor '((1 2) (3 4))))
(tensor-dtype (tensor '(1.0 2.0)))
]

@section[#:tag "welcome-two-surfaces"]{Two surfaces, two casings}

rktorch mirrors PyTorch's split between @emph{functional} operations and
@emph{layer} constructors, and uses case to keep them apart:

@itemlist[

 @item{lowercase names on @racketmodname[torch] are functions on tensors ---
 @racket[relu], @racket[matmul], @racket[sum] --- mirroring @tt{torch.relu}
 and friends.}

 @item{PascalCase names on @racketmodname[torch/nn] are layer constructors
 --- @racket[Linear], @racket[Conv2d], @racket[Sequential] --- mirroring
 @tt{torch.nn}'s classes.}

]

Because the casings differ, @racket[(require torch torch/nn)] never
collides: no prefix and no @racket[except-in] are needed.

@section[#:tag "welcome-shadowing"]{Shadowed names stay safe}

A handful of tensor operations share names with @racketmodname[racket/base]
--- @racket[exp], @racket[log], @racket[sqrt], @racket[max], @racket[min],
@racket[sort], and the arithmetic operators. rktorch provides these as
@emph{generic} operations: a tensor argument goes to libtorch, and anything
else falls through to the binding @racketmodname[racket/base] would have
given you.

So requiring the library never breaks the numeric code already in your
module:

@torch-examples[
(+ 1 2)
(+ (tensor '(1 2 3)) (tensor '(10 20 30)))
(+ (tensor '(1 2 3)) 10)
]

The first is @racketmodname[racket/base]'s @racket[+] on two numbers. The
second dispatches to tensor addition. The third broadcasts the scalar, and
promotes to a float tensor on the way, exactly as PyTorch does.

@margin-note{When you write Scribble documentation that mentions these
names, the @racket[for-label] import needs
@racket[(except-in racket/base exp log sort sqrt max min + - * /)] so the
links resolve to the tensor operations. The manual's own
@tt{scribblings/common.rkt} does this once for every chapter.}

@section[#:tag "welcome-next"]{Where to go next}

@secref["tensors"] covers building tensors and operating on them.
@secref["autograd"] introduces the gradients that make training possible,
@secref["layers"] the models that hold parameters, and
@secref["training"] the loop that fits one to data.
