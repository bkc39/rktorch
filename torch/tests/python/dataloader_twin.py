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



# with workers: worker k reseeds its global stream with base_seed + k, and
# the iterator primes prefetch_factor * num_workers requests
def drawing_collate(items):
    return (torch.stack([s[0] for s in items]),
            torch.empty((), dtype=torch.int64).random_().item(),
            torch.randperm(4).tolist(),
            torch.multinomial(torch.ones(5), 2).tolist(),
            torch.rand(3).tolist(),
            torch.randn(3).tolist())


def record(b):
    return [b[0].tolist()] + list(b[1:])


def workers_loader(gen, nw, pf, batch, drop_last, shuffle):
    return DataLoader(TensorDataset(torch.arange(N)), batch_size=batch,
                      shuffle=shuffle, drop_last=drop_last, generator=gen,
                      num_workers=nw, prefetch_factor=pf,
                      collate_fn=drawing_collate,
                      multiprocessing_context="fork")


def workers_case(nw, pf, batch, drop_last, shuffle, take):
    gen = torch.Generator().manual_seed(11)
    loader = workers_loader(gen, nw, pf, batch, drop_last, shuffle)
    it = iter(loader)
    first = [record(next(it))
             for _ in range(len(loader) if take == "all" else take)]
    then = torch.randperm(N, generator=gen).tolist()
    epochs = [[record(b) for b in loader] for _ in range(EPOCHS)]
    del it
    return {"first": first, "then": then, "epochs": epochs}


workers = {}
for nw, pf in [(1, 1), (2, 2), (3, 1)]:
    for batch in [4, 5]:
        for drop_last in [False, True]:
            for shuffle in [False, True]:
                for take in [0, 1, "all"]:
                    key = (f"w{nw}p{pf}b{batch}{'d' if drop_last else ''}"
                           f"{'s' if shuffle else ''}t{take}")
                    workers[key] = workers_case(nw, pf, batch, drop_last,
                                                shuffle, take)

torch.manual_seed(3)
for _ in range(EPOCHS):
    for _ in workers_loader(None, 2, 2, BATCH, False, True):
        pass
after_global_workers = torch.randn(3).tolist()

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
    "workers": workers,
    "after_global_workers": after_global_workers,
    "losses": losses,
    "params": [float(v) for v in torch.cat([p.detach().flatten()
                                           for p in model.parameters()]).tolist()],
}))
