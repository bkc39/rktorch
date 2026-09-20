Right — the latent already followed `device` and the two targets did not,
so a direct `train-step` on a model that is not on the default device
mixed operands. Both are built with `#:device device` now, and the
example's prose says so. Fixed in e817c34.
