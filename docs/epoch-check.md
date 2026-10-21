# The epoch check

CI trains each example for two to five steps on fixtures, on a CPU: it shows
the code runs, not that it still learns. The epoch check trains every
architecture the examples cover for a few epochs on the real data, on the lab
host's GPU, and compares each headline with the record below.

Run it at the top commit of a stack before marking the stack ready for review,
and before a release. Its table goes in the top pull request's review packet.

```bash
nix develop .#cuda -c scripts/epoch-check                            # every example, about eight minutes
nix develop .#cuda -c scripts/epoch-check 12-char-rnn 13-translation # a subset while iterating
```

Each runner in `examples/test/` runs as its own process, from
`.scratch/epoch-check/<example>/`, which holds its log and anything it writes;
`EPOCH_CHECK_OUT` moves that directory, as when comparing two checkouts.
Datasets and weights download once into the loaders' caches; set the
`RKTORCH_*_DIR` variables to share one cache between checkouts.

| verdict | meaning |
|---|---|
| `ok` | within the tolerance of the record |
| `REGRESSION` | worse than the record by more than the tolerance; the script exits 1 |
| `moved` | better by more than the tolerance |
| `FAILED` | the runner exited non-zero or printed no headline; the script exits 1 |

A `REGRESSION` or `FAILED` row blocks the stack until it is explained. A change
that moves a number on purpose, such as a new hyperparameter or a fixed bug,
updates the record in the same pull request and says why in its packet. A new
example adds its row.

## Record

Taken on 2026-10-14 at 8650a21, the top of stack A (#292) on master 1963034,
on an RTX 3090 Ti with libtorch 2.14.0 (cu130). Two runs there, one at #272's
head and one on master put cuDNN's run-to-run spread at 0.013 for 09's
accuracy, 0.025 for 10's generator loss and under 0.001 for the rest; each
tolerance sits above that with room to spare. An A/B at 0651574 differed only
where #229, #249 and #250 changed the code on purpose. The 17-workers row,
from one run, was taken on 2026-10-20 at 452e8cb (#168), on the same host.

The script reads the table below: the headline is the last number on the last
log line that `pattern` matches.

| example | budget | headline | pattern | better | record | tolerance |
|---|---|---|---|---|---|---|
| 04-mlp | default | MLP, loss after 5 SGD steps | `losses: .*` | lower | 1.2586 | 0.01 |
| 05-mnist | EPOCHS=1 | CNN, test accuracy | `test acc [0-9.]+` | higher | 0.970 | 0.01 |
| 06-gpt | STEPS=500 | GPT, loss at step 400 | `loss [0-9.]+` | lower | 2.269 | 0.05 |
| 07-asr | EPOCHS=2 | transformer ASR with a CTC head, mean hybrid loss | `mean loss [0-9.]+` | lower | 2.438 | 0.05 |
| 08-diffusion | EPOCHS=1 | DDPM UNet, mean epsilon-MSE | `mean loss [0-9.]+` | lower | 0.0700 | 0.002 |
| 09-resnet | EPOCHS=2 | CIFAR ResNet-18, test accuracy | `test acc [0-9.]+` | higher | 0.777 | 0.02 |
| 10-dcgan | EPOCHS=1 | DCGAN, generator loss | `g [0-9.]+` | lower | 2.62 | 0.15 |
| 11-vae | EPOCHS=2 | VAE, mean loss | `loss [0-9.]+` | lower | 121.43 | 2 |
| 12-char-rnn | EPOCHS=5 | LSTM, mean loss | `mean loss [0-9.]+` | lower | 2.0009 | 0.04 |
| 13-translation | EPOCHS=5 | GRU seq2seq, held-out token error rate | `pairs: [0-9.]+` | lower | 0.5029 | 0.01 |
| 14-imagenet | default | pretrained ResNet-18, percent for rapeseed on hedge mustard | `[0-9.]+%  rapeseed` | higher | 61.4 | 0.2 |
| 15-finetune | default | fine-tuned ResNet-18, validation accuracy | `val acc [0-9.]+` | higher | 0.928 | 0.03 |
| 16-style-transfer | default | VGG-16 style transfer, style loss at step 100 | `step 100: style [0-9.e-]+` | lower | 5.53e-05 | 1.5e-05 |
| 17-workers | default | CNN on Racket-decoded images, four loader workers, test accuracy | `test acc [0-9.]+` | higher | 0.987 | 0.01 |

## Not covered

- `resnet34` and `resnet50` (the bottleneck block): no example runs them.
- MPS: the record is CUDA's. A Mac run is read against it by eye.
