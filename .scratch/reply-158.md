Done in the follow-up commit. `define-layer` had nowhere to put this: `#:contract` reaches the constructor only, and `torch/private/contract.rkt` refuses a contract on a private helper, which is why `normalize` was checking rank by hand (and why `UNet` raises its own error and `Upsample` leans on `upsample-nearest2d`, blaming `diffusion.rkt` rather than the caller).

So a forward formal can now carry one, in the `[id : contract]` notation #99 proposes for `#:init`:

```racket
#:forward ([x : image-batch/c])
```

`BatchNorm2d` declares `image-batch/c`; `BatchNorm1d` declares a new `feature-batch/c` (`[N C]` or `[N C L]`), beside `image-batch/c` in `foreign/contracts.rkt`. A wrong rank now reads

```
BatchNorm2d: contract violation
  expected: image-batch
  given: tensor([1., 1.])
  in: the 1st argument of (-> image-batch any)
  contract from: BatchNorm2d
  blaming: caller
```

Three notes on the implementation:

- The checker is built once where the layer is defined, not once per call, so a forward pays one flat check; the cost table in AGENTS.md is why.
- The blamed party is the label `caller`, not the importing module. A layer's forward has no module boundary of its own, so there is no importing module to name at the point the wrapper is built. It no longer misblames `torch/nn/batch-norm.rkt`, which was the actual defect.
- A bare formal still accepts anything, so no other layer changed, and the arity error from #18 is untouched (`forward-arity-test` still passes).

`UNet`'s hand-rolled guard and `Upsample`'s misplaced blame can move onto the same clause; I would rather do that in its own PR than widen this one, so I have left them alone here.
