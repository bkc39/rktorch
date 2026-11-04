#lang scribble/manual

@(require (only-in racket/string string-join)
          "common.rkt"
          torch/scribblings/labels
          (only-in torch/train/private/hooks
                   epoch-log epoch-only-log hook-batch? hook-form hook-formals
                   hook-log hook-name hook-order hook-sides hooks step-log))

@title[#:tag "train-hooks"]{Hooks}

Callbacks are C and the trainer is T. A hook defined by both runs the
callbacks first, in order, then the trainer, except where the table says
otherwise. Arguments follow the run; @tt{batch…} is the batch values.
@racket[log!] gives where a hook may log and its default: @tt{step} and
@tt{epoch} allow both with that default, @tt{epoch only} allows only per
epoch.

@(define (who h)
   (define sides (hook-sides h))
   (cond
     [(not (memq 'trainer sides)) "C"]
     [(not (memq 'callback sides)) "T"]
     [(eq? (hook-order h) 'trainer-first) "T, C"]
     [(eq? (hook-order h) 'by-kind) "C plain, T, C monitor and checkpoint"]
     [else "C, T"]))
@(define (arguments h)
   (define names (map symbol->string (hook-formals h)))
   (string-join (if (hook-batch? h) (append names (list "batch…")) names)))
@(define (log-cell h)
   (define rule (hook-log h))
   (cond
     [(equal? rule step-log) "step"]
     [(equal? rule epoch-log) "epoch"]
     [(equal? rule epoch-only-log) "epoch only"]
     [else "no"]))
@(define (defined-by h)
   (if (hook-form h)
       (racketidfont (symbol->string (hook-form h)))
       (tt "define-hook")))
@(define (hook-row h)
   (list (tt (symbol->string (hook-name h))) (defined-by h) (who h)
         (arguments h) (log-cell h)))

@tabular[#:sep @hspace[2]
         #:style 'boxed
         (cons (list @bold{hook} @bold{defined by} @bold{who}
                     @bold{arguments} @bold{@racket[log!]})
               (map hook-row hooks))]

The order of one @racket[fit], after Lightning 2.6.5, with
@italic{A} the @racket[#:accumulate-grad-batches]:

@itemlist[
 @item{@tt{configure-callbacks}; @tt{setup}; the device move;
       @tt{configure-optimizers}; @tt{on-fit-start}.}
 @item{With @racket[#:val] and sanity steps: @tt{on-sanity-check-start},
       a validation loop whose metrics are discarded,
       @tt{on-sanity-check-end}.}
 @item{As in Lightning, @racket[fit] leaves each layer's mode as the caller
       set it. A trainer with a layer in eval mode warns once, before
       @tt{on-train-start}, on the @racket['rktorch-train] topic and on
       @racket[(current-error-port)], where Lightning's warning is printed too.}
 @item{@tt{on-train-start}, then per epoch @tt{on-train-epoch-start} and per
       batch: the transfer hooks, @tt{on-train-batch-start}, the training
       step, @tt{on-before-zero-grad} on a window's first batch,
       @tt{on-before-backward} with the loss ÷ @italic{A},
       @tt{on-after-backward}, @tt{on-before-optimizer-step} at a window's
       end, the schedulers, @tt{on-train-batch-end}, and a validation loop
       when one is due.}
 @item{A @racket[#f] loss skips that batch's @tt{on-before-backward},
       backward pass and @tt{on-after-backward}. After @racket[skip-step!],
       @tt{on-before-optimizer-step} and the step do not run, unless the
       hook is the caller, in which case only the step is dropped. Either
       way @racket[run-global-step] and the per-step schedulers advance at
       the window's end.}
 @item{@tt{on-train-epoch-end} after the epoch's validation loop, then
       @tt{on-train-end}, @tt{on-fit-end} and @tt{teardown}.}
 @item{A validation loop runs in eval mode with gradients off:
       @tt{on-validation-start}, @tt{on-validation-epoch-start}, per batch
       the transfer hooks, @tt{on-validation-batch-start}, the validation
       step and @tt{on-validation-batch-end}, then
       @tt{on-validation-epoch-end} and @tt{on-validation-end}; each layer
       then returns to the mode it had before the loop.}]
