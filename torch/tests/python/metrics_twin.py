"""torchmetrics 1.9.0's multiclass accuracy, confusion matrix and perplexity
on seeded inputs, with the plain torch accuracy formula beside them. The
Racket twin is metrics-parity-test.rkt."""
import json

import torch
import torchmetrics
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassConfusionMatrix)
from torchmetrics.functional.classification import (multiclass_accuracy,
                                                    multiclass_confusion_matrix)
from torchmetrics.functional.text import perplexity

from pretrained_common import packed

g = torch.Generator().manual_seed(0)
logits = torch.randn(257, 10, generator=g)
labels = torch.randint(0, 10, (257,), generator=g)
labels[::17] = -100
seq = torch.randn(4, 9, 50, generator=g)
tgt = torch.randint(0, 50, (4, 9), generator=g)
tgt[:, -2:] = -100
clean = torch.randint(0, 10, (257,), generator=g)
CUT = 100


def streamed(metric):
    metric.update(logits[:CUT], labels[:CUT])
    metric.update(logits[CUT:], labels[CUT:])
    return metric.compute()


out = {"version": torchmetrics.__version__,
       "logits": packed(logits), "labels": packed(labels),
       "seq": packed(seq), "tgt": packed(tgt),
       "acc": {str(k): multiclass_accuracy(logits, labels, 10, average="micro", top_k=k,
                                           ignore_index=-100).item()
               for k in (1, 3, 5)},
       "plain": ((logits.argmax(1) == labels)[labels != -100]).float().mean().item(),
       "clean": packed(clean),
       "acc_by_ignore": {str(ii): multiclass_accuracy(logits, clean, 10, average="micro",
                                                      ignore_index=ii).item()
                         for ii in (None, 0)},
       "cm_by_ignore": {str(ii): packed(multiclass_confusion_matrix(logits, clean, 10,
                                                                     ignore_index=ii))
                        for ii in (None, 0)},
       "acc_indices": multiclass_accuracy(logits.argmax(1), labels, 10, average="micro",
                                          ignore_index=-100).item(),
       "cut": CUT,
       "acc_streamed": {str(k): streamed(MulticlassAccuracy(10, average="micro", top_k=k,
                                                            ignore_index=-100)).item()
                        for k in (1, 3)},
       "cm_streamed": packed(streamed(MulticlassConfusionMatrix(10, ignore_index=-100))),
       "cm": packed(multiclass_confusion_matrix(logits, labels, 10, ignore_index=-100)),
       "ppl": perplexity(seq, tgt, ignore_index=-100).item()}
print(json.dumps(out))
