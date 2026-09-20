Right — the rule in AGENTS.md is decidable and this one falls on the
`checked` side: `scheduler.rkt` is a module under `torch/nn/` that imports
`optimizer?`, so the plain binding stays for it and the facade takes the
contracted one. `optim.rkt` now ends with

```racket
(module+ checked
  (provide (contract-out [optimizer? (-> any/c boolean?)])))
```

and `nn.rkt` requires `(submod "nn/optim.rkt" checked)` beside the
`only-in`, as it already does for `buffer.rkt`, `layer.rkt`, `init.rkt`
and `parameter.rkt`. Fixed in 5f93371.
