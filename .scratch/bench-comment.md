DDPM UNet (the default 35.7M-parameter net) training step on the 3090 Ti, Adam, median of 8 steps after 4 warm-up, each arm in its own process:

| arm | batch | ms/step | ms/image | peak allocated |
|---|---|---|---|---|
| float32 | 128 | 251 | 1.96 | 9714 MiB |
| bfloat16 autocast | 128 | 182 | 1.42 | 9008 MiB |
| bfloat16 autocast | 256 | 313 | 1.22 | 16012 MiB |

Autocast buys 1.4x per step at the same batch and 1.6x per image at batch 256. Peak memory drops only 7%: the parameters, gradients, Adam moments and the GroupNorm activations stay float32 by design, and autocast keeps a float32-to-bfloat16 cast of every weight in its cache for the extent. Measured with `.scratch/autocast-bench.rkt` (arm via `ARM=fp32|bf16`, batch via `BATCH=`).
