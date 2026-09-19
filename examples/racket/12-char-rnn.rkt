#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sort sqrt max min length + - * /)
                     torch torch/nn))

@section[#:tag "ex-char-rnn"]{A character-level LSTM on Heart of Darkness}

The recurrent counterpart of the char-GPT: the same novella, the same
next-character objective, but the context lives in an @racket[LSTM]'s state
instead of an attention window. It follows Karpathy's char-rnn, which is also
what ocaml-torch's @tt{char_rnn} example reproduces: embed each character,
run the sequence through a stack of LSTM layers, project every hidden state
back to vocabulary logits.

Two things differ from the transformer in practice. Training needs
@racket[clip-grad-norm!], because a gradient carried back through every step
of a sequence can blow up. And generation is cheap: the state summarises
everything read so far, so each new character costs one step, not a pass over
the whole context.

@chunk[<r12-require>
(require racket/runtime-path
         (only-in racket/file file->string)
         torch torch/nn
         (only-in torch/data/text
                  contiguous-blocks
                  decode
                  encode
                  load-heart-of-darkness
                  load-text-fixture
                  text->vocab))]

@chunk[<r12-provide>
(provide char-rnn pick-device load-excerpt run-example train-excerpt
         train-novel sample)]

@bold{The model.} Token ids index a learned @racket[Embedding]; the
@racket[LSTM] reads the embedded sequence batch-first, @tt{[B, T, C]}; a
@racket[Linear] head maps each of its outputs to logits. The forward takes the
recurrent state as a second argument, @racket[#f] to start from zeros or the
@racket[(list h c)] a previous call answered, and answers the logits together
with the new state. Training passes @racket[#f] for every batch; sampling
threads the state from one character to the next. The layers are declared in
the order the Python twin declares them, which is the order their initial
weights are drawn in.

@chunk[<r12-model>
(define-layer char-rnn (embed lstm head)
  #:init (vocab-size
          #:n-embd [n-embd 32]
          #:hidden [hidden 64]
          #:num-layers [num-layers 1]
          #:dropout [dropout 0.0])
  (set! embed (Embedding vocab-size n-embd))
  (set! lstm (LSTM n-embd hidden
                   #:num-layers num-layers
                   #:dropout dropout
                   #:batch-first? #t))
  (set! head (Linear hidden vocab-size))
  #:forward (idx state)
  (define-values (out h c)
    (if state
        (lstm (embed idx) (car state) (cadr state))
        (lstm (embed idx))))
  (values (head out) (list h c)))]

@bold{The loss.} Every position predicts its successor, so the @tt{[B, T, V]}
logits and @tt{[B, T]} targets flatten to one @tt{[B*T]}-row classification.
@racket[nll-loss] over @racket[log-softmax] is @racket[cross-entropy] spelled
out; it is written this way because the translation capstone needs the two
halves apart.

@chunk[<r12-loss>
(define (next-char-loss net xs ys)
  (define-values (logits _state) (net xs #f))
  (define vocab-size (last-dim logits))
  (nll-loss (log-softmax (reshape logits -1 vocab-size) 1)
            (reshape ys -1)))

(define (last-dim t)
  (car (reverse (tensor-shape t))))]

@bold{One step.} Backward, clip, update. The clip sits between the backward
pass and the optimizer: it rescales all gradients by one factor so their joint
norm is at most @racket[max-norm], leaving their direction alone.

@chunk[<r12-step>
(define (train-step! net opt xs ys #:max-norm [max-norm 1.0])
  (zero-grads! opt)
  (define loss (next-char-loss net xs ys))
  (backward! loss)
  (clip-grad-norm! (parameters net) max-norm)
  (step! opt)
  loss)]

@chunk[<r12-device>
(define (pick-device)
  (accelerator-if-available))]

@bold{The deterministic core.} @racket[run-example] is what the test harness
and the PyTorch twin both drive: the committed 841-character fixture in
16-character blocks, full batch, five clipped @racket[adam] steps from seed 0.
Under that seed the embedding, the LSTM's four weight tensors and the
head start from PyTorch's values, and the losses and final parameters track
@tt{torch.optim.Adam} with @tt{clip_grad_norm_} within float tolerance.

@chunk[<r12-run>
(define fixture-block-size 16)

(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define text (load-text-fixture))
    (define vocab (text->vocab text))
    (define-values (xs ys)
      (contiguous-blocks (encode vocab text) fixture-block-size))
    (define net (char-rnn (vector-length vocab)))
    (define opt (adam (parameters net) #:lr 0.001))
    (define losses
      (for/list ([_ (in-range steps)])
        (item (train-step! net opt xs ys))))
    (values losses net vocab device)))]

@bold{Training for real.} @racket[train-on-text] is the epoch loop both
entry points share: batch-stride windows over the text's contiguous blocks,
the ragged tail dropped, the mean loss printed as it goes. Each batch starts
from a zero state, so the model learns to find its footing within one block;
a block of 64 characters is a sentence or two of Conrad, which is enough.
@racket[train-excerpt] runs it offline on the committed opening of Part I;
@racket[train-novel] downloads the whole novella (cached) and trains a
two-layer, 256-wide model with dropout between the layers, about a minute on
a GPU. The runner, @filepath{examples/test/12-char-rnn.rkt}, trains the novella
model and prints a sample; @envvar{EXCERPT} switches it to the offline
excerpt, and @envvar{EPOCHS}, @envvar{TEMPERATURE} and @envvar{SEED} override
the defaults.

@chunk[<r12-train>
(define-runtime-path excerpt-path "../data/heart-of-darkness-part-i.txt")

(define (load-excerpt)
  (file->string excerpt-path))

(define (train-on-text text net-for
                       #:epochs epochs #:batch batch #:block-size block-size
                       #:lr lr #:device device #:log-every log-every)
  (with-default-device device
    (manual-seed! 0)
    (define vocab (text->vocab text))
    (define-values (xs ys)
      (contiguous-blocks (encode vocab text) block-size))
    (define n (car (tensor-shape xs)))
    (unless (<= batch n)
      (error 'train-on-text "batch ~a exceeds the text's ~a blocks" batch n))
    (define net (net-for (vector-length vocab)))
    (define opt (adam (parameters net) #:lr lr))
    (for ([epoch (in-range 1 (add1 epochs))])
      (define-values (total steps)
        (for/fold ([total 0.0] [steps 0])
                  ([start (in-range 0 (add1 (- n batch)) batch)])
          (define loss
            (train-step! net opt
                         (narrow xs 0 start batch) (narrow ys 0 start batch)
                         #:max-norm 5.0))
          (values (+ total (item loss)) (add1 steps))))
      (when (zero? (modulo epoch log-every))
        (printf "epoch ~a/~a: mean loss ~a\n" epoch epochs (/ total steps))))
    (values net vocab)))

(define (train-excerpt #:epochs [epochs 40] #:batch [batch 32]
                       #:block-size [block-size 64]
                       #:device [device (pick-device)]
                       #:log-every [log-every 5])
  (train-on-text (load-excerpt)
                 (lambda (vocab-size)
                   (char-rnn vocab-size #:n-embd 64 #:hidden 128))
                 #:epochs epochs #:batch batch #:block-size block-size
                 #:lr 0.003 #:device device #:log-every log-every))

(define (train-novel #:epochs [epochs 30] #:batch [batch 64]
                     #:block-size [block-size 64]
                     #:device [device (pick-device)]
                     #:log-every [log-every 1])
  (train-on-text (load-heart-of-darkness)
                 (lambda (vocab-size)
                   (char-rnn vocab-size #:n-embd 64 #:hidden 256
                             #:num-layers 2 #:dropout 0.2))
                 #:epochs epochs #:batch batch #:block-size block-size
                 #:lr 0.002 #:device device #:log-every log-every))]

@bold{Sampling.} The prompt goes through the model once, which leaves the
state primed and the last position's logits predicting the first new
character. From there each step divides the logits by the
@racket[temperature], turns them into probabilities with @racket[softmax], and
draws one index with @racket[multinomial]; that index, as a @tt{[1, 1]} batch,
is the next input, alongside the state the previous step answered. A
temperature below one sharpens the distribution toward the likeliest
characters, above one flattens it; as it approaches zero the draw approaches
the @racket[argmax] the GPT example uses. The draws come from the global
stream, so @racket[manual-seed!] makes a sample repeatable. Everything runs
where the weights are, under @racket[in-eval-mode] and @racket[with-no-grad],
and only the chosen index crosses to the host each step.

@chunk[<r12-sample>
(define (sample net vocab prompt
                #:steps [steps 256]
                #:temperature [temperature 0.8])
  (when (zero? (string-length prompt))
    (error 'sample "prompt must be non-empty"))
  (unless (positive? temperature)
    (error 'sample "temperature must be positive, got ~a" temperature))
  (define device (tensor-device (car (parameters net))))
  (define (last-step-logits logits)
    (define seq-len (cadr (tensor-shape logits)))
    (reshape (narrow logits 1 (- seq-len 1) 1) 1 -1))
  (define (draw logits)
    (multinomial (softmax (/ logits temperature) -1) 1))
  (in-eval-mode net
    (with-no-grad
      (define prompt-ids
        (to-device (reshape (encode vocab prompt) 1 -1) device))
      (define-values (prompt-logits primed) (net prompt-ids #f))
      (define-values (drawn _logits _state)
        (for/fold ([drawn '()]
                   [logits (last-step-logits prompt-logits)]
                   [state primed])
                  ([_ (in-range steps)])
          (define next (draw logits))
          (define-values (next-logits next-state) (net next state))
          (values (cons (inexact->exact (item next)) drawn)
                  (last-step-logits next-logits)
                  next-state)))
      (string-append prompt (decode vocab (reverse drawn))))))]

@chunk[<*>
  <r12-require>
  <r12-provide>
  <r12-model>
  <r12-loss>
  <r12-step>
  <r12-device>
  <r12-run>
  <r12-train>
  <r12-sample>]
