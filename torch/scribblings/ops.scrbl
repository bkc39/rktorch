#lang scribble/manual

@(require "common.rkt"
          (for-label (except-in racket/base
                                abs cos exp log sin sort sqrt max min length
                                + - * /)
                     racket/contract
                     torch
                     (only-in torch/data/loader dataloader define-dataset)
                     (only-in torch/nn MaxPool2d cross-entropy)))

@title{Operations on tensors}

@defmodule[torch #:link-target? #f]

@section{Shape}

@defproc[(reshape [t tensor?] [dim exact-integer?] ...) tensor?]{
A view of @racket[t] with the given shape, which must have the same number
of elements. One dimension may be @racket[-1], and is solved for.

@torch-examples[
(reshape (arange 6) 2 3)
(shape (reshape (arange 6) -1 3))
]}

@defproc[(transpose [t tensor?] [dim1 exact-integer?] [dim2 exact-integer?])
         tensor?]{
A view with the two named axes exchanged.

@torch-examples[(shape (transpose (zeros 2 3) 0 1))]}

@defidform[t]{
A terse alias for @racket[transpose], taking the same three arguments. See
also @racket[T], which reverses every axis.}

@defproc[(T [x tensor?]) tensor?]{
A view with every dimension in reverse order, matching Python's @tt{x.T}. A
matrix of shape @tt{[M, N]} becomes @tt{[N, M]}; a tensor of shape
@tt{[B, H, S, D]} becomes @tt{[D, S, H, B]}. Scalars and vectors keep their
shapes. Storage is shared with @racket[x], and gradients propagate through
the view.

@torch-examples[
(T (tensor '((1 2 3) (4 5 6))))
]

For batched attention keys, @racket[(transpose keys -2 -1)] swaps only the
last two dimensions, as Python's @tt{keys.mT} does, where @racket[T]
reverses every axis. PyTorch deprecates @tt{.T} for tensors whose rank is
not two; @racket[T] supports every rank without a warning.}

@section{Arithmetic}

@defproc[(mul [a (or/c tensor? real?)]
              [b (if (tensor? a) (or/c tensor? real?) tensor?)])
         tensor?]{
Elementwise product. A real operand broadcasts across the tensor; at least
one of the two must be a tensor.

A scalar crosses the FFI boundary as a C double, so an integer tensor
combined with an integer scalar comes back @racket['float32] where PyTorch
would keep the integer dtype.

@torch-examples[(tensor->list (mul (tensor '(1.0 2.0)) 3.0))]}

@deftogether[(@defproc[(add [a (or/c tensor? real?)]
                            [b (if (tensor? a) (or/c tensor? real?) tensor?)])
                       tensor?]
              @defproc[(sub [a (or/c tensor? real?)]
                            [b (if (tensor? a) (or/c tensor? real?) tensor?)])
                       tensor?])]{
Elementwise sum and difference, with the same broadcasting, dtype and
at-least-one-tensor rules as @racket[mul]. @racket[+] and @racket[-] are
the operator spellings.

@torch-examples[
(add (tensor '(1.0 2.0)) 10)
(sub 10 (tensor '(1.0 2.0)))
]}

@defproc[(matmul [a tensor?] [b tensor?]) tensor?]{
Matrix product, following PyTorch's @tt{torch.matmul} broadcasting rules.

@torch-examples[
(matmul (tensor '((1.0 2.0) (3.0 4.0))) (tensor '((1.0 0.0) (0.0 1.0))))
]}

@defidform[|@|]{
The operator spelling of @racket[matmul], like Python's @tt{a @"@" b}.
Scribble reserves bare @litchar["@"], so it appears here as
@racket[|@|]; in ordinary code it is written @litchar["@"].}

@section{Threading}

@deftogether[(@defidform[~>] @defidform[~>>]
              @defidform[lambda~>] @defidform[lambda~>>])]{
Re-exported from the @hyperlink["https://docs.racket-lang.org/threading/"]{
@tt{threading}} library, so they are in scope with @racketmodname[torch]
and need no separate import.

@torch-examples[(~> (arange 6) (reshape 2 3) sum item)]}

@section{Elementwise functions}

@defproc[(relu [t tensor?]) tensor?]{
The rectifier, @tt{max(0, x)} elementwise.

@torch-examples[(relu (tensor '(-1.0 0.0 2.0)))]}

@section{Reductions}

@defproc[(sum [t tensor?]) tensor?]{
Adds every element, answering a one-element tensor. Use @racket[item] to
get a Racket number back.

@torch-examples[(sum (tensor '((1.0 2.0) (3.0 4.0))))]}

@defproc[(mean [t tensor?]) tensor?]{
The arithmetic mean of every element, as a one-element tensor.

@torch-examples[(mean (tensor '(1.0 2.0 3.0)))]}

@margin-note{Neither takes an axis argument yet: both are whole-tensor
reductions, where PyTorch's @tt{sum} and @tt{mean} accept a @tt{dim}.}

@defidform[Σ]{
A terse alias for @racket[sum].

@torch-examples[(item (Σ (tensor '(1 2 3))))]}

@defproc[(argmax [t tensor?]
                 [dim (or/c exact-integer? #f) #f]
                 [#:keepdim keepdim boolean? #f])
         tensor?]{
The index of the largest element: over the flattened tensor without
@racket[dim], or along @racket[dim], where @racket[#:keepdim] keeps that
axis with length one. An int64 tensor.

@torch-examples[
(argmax (tensor '((1.0 9.0 3.0) (7.0 2.0 4.0))) 1)
]}

@section{Comparison}

Each answers a @racket['bool] tensor, elementwise, the second operand a
tensor or a real that broadcasts. The mask feeds @racket[where],
@racket[masked-fill], or a @racket[sum] that counts the hits.

@deftogether[(@defproc[(eq [a tensor?] [b (or/c tensor? real?)]) tensor?]
              @defproc[(ne [a tensor?] [b (or/c tensor? real?)]) tensor?]
              @defproc[(lt [a tensor?] [b (or/c tensor? real?)]) tensor?]
              @defproc[(le [a tensor?] [b (or/c tensor? real?)]) tensor?]
              @defproc[(gt [a tensor?] [b (or/c tensor? real?)]) tensor?]
              @defproc[(ge [a tensor?] [b (or/c tensor? real?)]) tensor?])]{
Equal, not equal, less than, less than or equal, greater than, greater
than or equal.

@torch-examples[
(eq (tensor '(1 2 3)) 2)
(item (sum (ge (tensor '(1 2 3)) 2)))
]}

@section{Selection}

@defproc[(narrow [t tensor?] [dim exact-integer?]
                 [start exact-integer?] [len exact-positive-integer?])
         tensor?]{
A view of @racket[len] entries along @racket[dim] starting at
@racket[start]: the slice @tt{t[start:start+len]} on that axis, sharing
storage.

@torch-examples[(narrow (arange 6) 0 2 3)]}

@defproc[(select [t tensor?] [dim exact-integer?] [index exact-integer?])
         tensor?]{
The slice at @racket[index] along @racket[dim], with that axis removed,
as @tt{t.select(dim, index)}; a view.

@torch-examples[(select (tensor '((1 2) (3 4))) 0 1)]}

@defform[(ref t spec ...)
         #:grammar
         ([spec index-expr
                (code:line :)
                (code:line ..)
                (code:line _)
                (: stop)
                (: start stop)
                (: start stop step)
                (:~ start)
                (:~ start step)])]{
Python's indexing and slicing, one @racket[spec] per axis: an integer
index, @racket[:] for a whole axis, @racket[(: start stop step)] for a
slice with @racket[_] leaving an end open, @racket[(:~ start)] for a slice
to the end, and @racket[..] for however many whole axes lie between. Fully
indexed, it answers the element as a number or boolean; otherwise a view.
The tokens are matched by name, so they need no import.

@torch-examples[
(ref (tensor '((1 2 3) (4 5 6))) 1 2)
(ref (tensor '((1 2 3) (4 5 6))) : (: 1 3))
]}

@defform[(ref! t value spec ...)]{
Writes @racket[value], a tensor or a real, into the selection @racket[spec ...]
names, in place.}

@defproc[(index-select [t tensor?] [dim exact-integer?] [indices tensor?])
         tensor?]{
The slices at each of @racket[indices], an int64 vector, along
@racket[dim], gathered into a new tensor in that order.

@torch-examples[(index-select (arange 5) 0 (tensor '(4 0)))]}

@defproc*[([(take [t tensor?] [indices (or/c tensor? (listof exact-integer?)
                                             (vectorof exact-integer?))])
            tensor?]
           [(take [lst list?] [n exact-nonnegative-integer?]) list?])]{
On a tensor, the elements at @racket[indices] into the flattened tensor,
as @tt{torch.take}. On a list, @racketmodname[racket/list]'s first-@racket[n]
elements, so requiring @racketmodname[torch] does not break list code.}

@defproc*[([(where [c tensor?]) (listof tensor?)]
           [(where [c tensor?] [a (or/c tensor? real?)] [b (or/c tensor? real?)])
            tensor?])]{
With three arguments, @racket[a] where the @racket['bool] tensor
@racket[c] holds and @racket[b] elsewhere, broadcasting all three. With
one, the indices where @racket[c] holds, one int64 tensor per dimension.

@torch-examples[
(where (eq (tensor '(1 0 1)) 1) (tensor '(1 2 3)) 0)
]}

@defproc[(masked-fill [t tensor?] [mask tensor?] [value real?]) tensor?]{
A copy of @racket[t] with @racket[value] wherever the @racket['bool]
@racket[mask] holds. The causal mask of an attention block is
@racket[(masked-fill scores (eq (tril (ones n n)) 0) -inf.0)].}

@defproc[(tril [t tensor?] [diagonal exact-integer? 0]) tensor?]{
The lower triangle of a matrix, or of each matrix in a batch, with the
rest zeroed; @racket[diagonal] shifts the boundary up or down.

@torch-examples[(tril (ones 3 3))]}

@section{Joining and reshaping}

@deftogether[(@defproc[(cat [ts (non-empty-listof tensor?)] [dim exact-integer? 0])
                       tensor?]
              @defproc[(stack [ts (non-empty-listof tensor?)] [dim exact-integer? 0])
                       tensor?])]{
@racket[cat] joins tensors along an existing axis; @racket[stack] joins
them along a new one, so every input must have the same shape.

@torch-examples[
(cat (list (tensor '(1 2)) (tensor '(3))))
(stack (list (tensor '(1 2)) (tensor '(3 4))))
]}

@defproc[(unsqueeze [t tensor?] [dim exact-integer?]) tensor?]{
A view with a length-one axis inserted at @racket[dim].

@torch-examples[(shape (unsqueeze (arange 3) 0))]}

@defproc*[([(flatten [t tensor?] [start-dim exact-integer? 0]
                     [end-dim exact-integer? -1])
            tensor?]
           [(flatten [v any/c]) list?])]{
On a tensor, a view with the axes from @racket[start-dim] through
@racket[end-dim] merged into one, so @racket[(flatten x 1)] turns a batch
of feature maps into a batch of vectors. On anything else,
@racketmodname[racket/list]'s @racket[flatten].

@torch-examples[(shape (flatten (zeros 2 3 4) 1))]}

@section{Activations}

@deftogether[(@defproc[(sigmoid [t tensor?]) tensor?]
              @defproc[(gelu [t tensor?]) tensor?]
              @defproc[(leaky-relu [t tensor?]
                                   [#:negative-slope slope real? 0.01])
                       tensor?])]{
The logistic function, the Gaussian error linear unit, and the rectifier
with a small slope for negative inputs. @racket[tanh] is generic over
tensors and reals, like @racket[exp].

@torch-examples[(sigmoid (tensor '(0.0)))]}

@section{Softmax, ordering and sampling}

@deftogether[(@defproc[(softmax [t tensor?] [dim exact-integer?]) tensor?]
              @defproc[(log-softmax [t tensor?] [dim exact-integer?]) tensor?])]{
The softmax along @racket[dim], and its logarithm computed stably.
@racket[cross-entropy] takes raw logits and applies the latter itself.

@torch-examples[(softmax (tensor '(1.0 2.0 3.0)) 0)]}

@defproc[(topk [t tensor?] [k exact-positive-integer?]
               [#:dim dim exact-integer? -1]
               [#:largest? largest? boolean? #t]
               [#:sorted? sorted? boolean? #t])
         (values tensor? tensor?)]{
The @racket[k] largest (or smallest) entries along @racket[dim] and their
indices, as two values.

@torch-examples[(topk (tensor '(3.0 1.0 2.0)) 2)]}

@defproc[(argsort [t tensor?]
                  [#:dim dim exact-integer? -1]
                  [#:descending? descending? boolean? #f])
         tensor?]{
The indices that would sort @racket[t] along @racket[dim].}

@defproc[(multinomial [probabilities tensor?] [n exact-positive-integer?]
                      [#:replacement? replacement? boolean? #f]
                      [#:generator generator (or/c generator? #f) #f])
         tensor?]{
@racket[n] indices drawn from each row of @racket[probabilities], which
need not sum to one, from @racket[generator] or the global stream;
seeded CPU draws match PyTorch's. Sampling the next character of a
language model is @racket[(multinomial (softmax logits -1) 1)].}

@section{Spatial}

@defproc[(max-pool2d [t tensor?]
                     [kernel (or/c exact-positive-integer?
                                   (list/c exact-positive-integer?
                                           exact-positive-integer?))]
                     [#:stride stride (or/c #f exact-positive-integer?
                                            (list/c exact-positive-integer?
                                                    exact-positive-integer?))
                               #f]
                     [#:padding padding (or/c exact-nonnegative-integer?
                                              (list/c exact-nonnegative-integer?
                                                      exact-nonnegative-integer?))
                                0]
                     [#:dilation dilation (or/c exact-positive-integer?
                                                (list/c exact-positive-integer?
                                                        exact-positive-integer?))
                                 1]
                     [#:ceil-mode ceil-mode boolean? #f])
         tensor?]{
Max pooling over an @tt{[N C H W]} batch; the stride defaults to the
kernel size. The layer form is @racket[MaxPool2d].

@torch-examples[(shape (max-pool2d (zeros 1 1 8 8) 2))]}

@defproc[(upsample-nearest2d [t tensor?] [#:scale scale exact-positive-integer? 2])
         tensor?]{
Each pixel of an @tt{[N C H W]} batch repeated @racket[scale] times along
both spatial axes.

@torch-examples[(shape (upsample-nearest2d (zeros 1 1 4 4)))]}

@section{Length}

@defproc[(length [v sized?]) exact-nonnegative-integer?]{
Python's @tt{len}, shadowing @racketmodname[racket/base]'s @racket[length]
the way @racket[+] is shadowed: a list, vector, string or hash answers what
it always did, a tensor answers its first dimension, and a dataset or
loader answers its number of items or batches.

@torch-examples[
(length '(1 2 3))
(length (zeros 4 2))
]

A rank-zero tensor has no length, as @tt{len} of a 0-d tensor raises.}

@defproc[(sized? [v any/c]) boolean?]{
Recognises anything @racket[length] accepts.}

@defthing[gen:sized any/c]{
The generic behind @racket[length], with that one method. A structure
implements it with @racket[#:methods]; @racket[define-dataset] does so for
every dataset, which is what lets @racket[length] answer a
@racket[dataloader].}

@section{Shadowed names}

A few operations share a name with @racketmodname[racket/base]. Each is
generic: a tensor argument dispatches to libtorch, and anything else defers
to the binding @racketmodname[racket/base] provides, so requiring
@racketmodname[torch] never breaks numeric code that was already in the
module. @racket[length], above, is the same kind of generic.

@deftogether[(@defidform[abs] @defidform[cos] @defidform[exp]
              @defidform[log] @defidform[sin] @defidform[sqrt]
              @defidform[tanh] @defidform[max] @defidform[min]
              @defidform[sort])]{
Generic over tensors and the values @racketmodname[racket/base] accepts.}

@deftogether[(@defidform[+] @defidform[-] @defidform[*] @defidform[/])]{
The arithmetic operators, provided as renames rather than contracted
wrappers so the numeric path costs nothing. Numeric operands take
@racketmodname[racket/base]'s path; a tensor operand on either side
dispatches to the tensor operation, and a chain folds left.

@torch-examples[
(+ 1 2)
(+ (tensor '(1 2 3)) (tensor '(10 20 30)))
(+ (tensor '(1 2 3)) 10)
]}
