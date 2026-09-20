Headline run on the RTX 3090 Ti: `EPOCHS=30 raco test -s main examples/test/09-resnet.rkt`, ResNet-18 at base 64, batch 128, random crop and flip on the device, SGD with Nesterov momentum 0.9 and weight decay 5e-4 under a one-cycle schedule peaking at 0.1, the forward in bfloat16 under autocast. Thirty epochs in 422 s, about 14 s per epoch including the test pass.

| epoch | test acc | epoch | test acc | epoch | test acc |
|---|---|---|---|---|---|
| 1 | 0.586 | 11 | 0.834 | 21 | 0.894 |
| 2 | 0.687 | 12 | 0.751 | 22 | 0.900 |
| 3 | 0.684 | 13 | 0.839 | 23 | 0.906 |
| 4 | 0.700 | 14 | 0.851 | 24 | 0.917 |
| 5 | 0.781 | 15 | 0.824 | 25 | 0.930 |
| 6 | 0.756 | 16 | 0.868 | 26 | 0.932 |
| 7 | 0.787 | 17 | 0.880 | 27 | 0.936 |
| 8 | 0.788 | 18 | 0.868 | 28 | 0.937 |
| 9 | 0.755 | 19 | 0.864 | 29 | 0.940 |
| 10 | 0.770 | 20 | 0.885 | 30 | **0.941** |

94.1% at epoch 30, above the 93% the leg set out for; the swings in the first half are the schedule's peak rate, and the last ten epochs are the anneal. Log at `~/cifar10-resnet/train.log`.
