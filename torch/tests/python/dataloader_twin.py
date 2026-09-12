"""DataLoader over a TensorDataset with a seeded generator, two epochs.

The batch index order is what the Racket loader must replay: it comes from
a real DataLoader, so it includes everything one epoch draws from the
generator. The losses come from a tiny linear model trained on those
batches, so parity covers the loop, not just the permutation.
"""
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

N, BATCH, EPOCHS = 10, 4, 2
xs = torch.arange(N * 3, dtype=torch.float32).reshape(N, 3) / N
ys = xs @ torch.ones(3, 1)

index_loader = DataLoader(TensorDataset(torch.arange(N)), batch_size=BATCH,
                          shuffle=True, generator=torch.Generator().manual_seed(7))
loader_order = [[[int(v) for v in b[0]] for b in index_loader]
                for _ in range(EPOCHS)]

# no generator: the draws come from the global stream and advance it
torch.manual_seed(3)
global_loader = DataLoader(TensorDataset(torch.arange(N)), batch_size=BATCH,
                           shuffle=True)
global_order = [[[int(v) for v in b[0]] for b in global_loader]
                for _ in range(EPOCHS)]
after_global = torch.randn(3).tolist()

# every iterator draws a base seed, shuffled or not, from the loader's
# generator or the global stream
g = torch.Generator().manual_seed(5)
plain = DataLoader(TensorDataset(torch.arange(N)), batch_size=BATCH, generator=g)
for _ in range(EPOCHS):
    for _ in plain:
        pass
unshuffled_then = torch.randperm(N, generator=g).tolist()
torch.manual_seed(3)
for _ in DataLoader(TensorDataset(torch.arange(N)), batch_size=BATCH):
    pass
after_global_plain = torch.randn(3).tolist()


# the trailing permutation is drawn once the first is used up: before a
# final partial batch, else only when the iterator is exhausted
def partial(batch, take):
    gen = torch.Generator().manual_seed(11)
    loader = DataLoader(TensorDataset(torch.arange(N)), batch_size=batch,
                        shuffle=True, generator=gen)
    epoch = iter(loader)
    first = [[int(v) for v in next(epoch)[0]]
             for _ in range(len(loader) if take == "all" else 1)]
    second = [[int(v) for v in b[0]] for b in loader]
    return {"first": first, "second": second,
            "then": torch.randperm(N, generator=gen).tolist()}


partial_orders = {"one_of_4": partial(4, "one"), "all_of_4": partial(4, "all"),
                  "all_of_5": partial(5, "all")}

torch.manual_seed(0)
model = torch.nn.Linear(3, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loader = DataLoader(TensorDataset(xs, ys), batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(7))
losses = []
for _ in range(EPOCHS):
    for xb, yb in loader:
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(float(loss))

print(json.dumps({
    "loader_order": loader_order,
    "global_order": global_order,
    "after_global": after_global,
    "unshuffled_then": unshuffled_then,
    "after_global_plain": after_global_plain,
    "partial_orders": partial_orders,
    "losses": losses,
    "params": [float(v) for v in torch.cat([p.detach().flatten()
                                           for p in model.parameters()]).tolist()],
}))
