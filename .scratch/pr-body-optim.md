Leg 2 of #152, the optimizer surface: the SGD family, RMSprop, and
learning-rate schedules toward #144. Stacked on #165 (leg 1); the base
retargets as the stack merges.

## What lands

- **sgd** takes `#:momentum`, `#:nesterov?` and `#:weight-decay` with
  `torch.optim.SGD`'s arithmetic (buffer copied from the first gradient,
  blended after; Nesterov one blend ahead, which needs a momentum, as a
  `->i` precondition; L2 decay added to the gradient). **adam** takes the
  same L2 `#:weight-decay`. **rmsprop** is `torch.optim.RMSprop`,
  uncentered, with its zero-initialised momentum buffer.
- **learning-rate** and **set-learning-rate!** over a new pair on
  `gen:optimizer`; every optimizer's rate is a mutable field.
- **Schedules** (`torch/nn/scheduler.rkt`): a schedule wraps an optimizer
  and implements `gen:optimizer`, so `step!` advances it and writes the
  closed-form rate; the rate for step 0 is written at construction, as in
  PyTorch. `step-lr`, `multi-step-lr`, `exponential-lr`,
  `cosine-annealing-lr`, `linear-lr` (warmup), `one-cycle-lr` (cosine
  phases, momentum left alone, an error past the cycle), `lambda-lr`.
  Chaining, sequencing and parameter groups stay open on #144.
- Manual section "Optimizers and schedules" in the layers chapter; AGENTS
  roster.

## Verification

- CPU (`.#ci`): nn, scheduler, to, define-layer, nn-contract, diffusion,
  convnet-smoke suites and the MLP, MNIST and GPT example harnesses (the
  existing Adam and SGD callers unchanged); `raco review`, `resyntax
  analyze` against master, scribble.
- GPU (`.#cuda`): python-cross with two new twins: `sgd_variants.py` trains
  the seeded MLP five steps under six configurations (momentum, Nesterov,
  weight decay, Adam with decay, RMSprop, RMSprop with momentum and decay)
  and pins losses and parameters; `schedulers.py` pins every shape's rates
  over twelve steps from `get_last_lr`, at 1e-9.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VDEpCNkMi2rmxjgRnCHhkp
