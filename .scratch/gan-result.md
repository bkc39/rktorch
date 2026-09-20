Headline runs on the RTX 3090 Ti, `OUT=~/mnist-generative`, both under two minutes end to end.

**DCGAN**, `EPOCHS=5`, mean losses per epoch (discriminator, generator): 0.348 / 2.95, 0.292 / 2.53, 0.286 / 2.79, 0.306 / 2.70, 0.395 / 2.51. The discriminator holds the upper hand at this size, as the reference does; the epoch-1 grid is blobs, and by epoch 5 the same hundred latents are crisp, mostly legible digits with the occasional malformed stroke a five-epoch DCGAN leaves.

**VAE**, `EPOCHS=10`, mean loss per epoch: 165.2, 121.4, 114.7, 111.7, 109.8, 108.7, 107.9, 107.2, 106.7, 106.2, the reference architecture's plateau. The epoch-10 grid is the soft, legible digits a linear VAE is known for.

Grids as PNG with an index: `http://100.90.250.112:8788/` on the tailnet, from `~/mnist-generative` (the PPMs are beside them). Log at `~/mnist-generative/train.log`.
