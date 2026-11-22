"""The probes of examples/python/19_dino.py, for probe-parity-test.rkt.

Ten-class Gaussian blobs in 24 dimensions, seeded 0, as fit, select and
test splits; the twin's stratified split of the CIFAR-10 fixture's labels
(and of the cached training archive's, when present); its standardiser,
fit_linear and probe_suite; sklearn's logistic regression and cosine kNN
on the twin-standardised float64 features; and torch.optim.LBFGS on the
2-D Rosenbrock function, on probe-fit-test.rkt's P14 quadratic, and on the
probe objective, each closure's loss recorded.
"""
import importlib.util
import json
import os
import tarfile

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from pretrained_common import packed

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "..", "..", "examples", "python")
FIXTURE = os.path.join(HERE, "..", "..", "vision", "fixtures",
                       "cifar10-256.bin")
RECORD = 3073


def dino():
    spec = importlib.util.spec_from_file_location(
        "dino", os.path.join(EXAMPLES, "19_dino.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


twin = dino()

g = torch.Generator().manual_seed(0)
centres = 0.85 * torch.randn(10, 24, generator=g)


def split(n):
    y = torch.randint(0, 10, (n,), generator=g)
    return centres[y] + torch.randn(n, 24, generator=g), y


(fit, fy), (sel, sy), (te, ty) = split(3000), split(600), split(1000)

with open(FIXTURE, "rb") as f:
    data = f.read()
labels256 = torch.tensor([data[i * RECORD] for i in range(256)])
twin.SELECT_PER_CLASS = 2
split256 = twin.stratified_split(labels256)


def cifar_train_labels():
    root = os.environ.get("RKTORCH_CIFAR10_DIR",
                          os.path.expanduser("~/.racket/rktorch/cifar10"))
    path = os.path.join(root, twin.ARCHIVE)
    if not os.path.exists(path):
        return None
    with tarfile.open(path, "r:gz") as tar:
        blobs = {os.path.basename(m.name): tar.extractfile(m).read()
                 for m in tar.getmembers() if m.name.endswith(".bin")}
    raw = b"".join(blobs[f"data_batch_{i}.bin"] for i in range(1, 6))
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8
                            ).reshape(-1, RECORD)[:, 0].long()


full = cifar_train_labels()
twin.SELECT_PER_CLASS = 500
full_split = None if full is None else [packed(i) for i in twin.stratified_split(full)]

fit_std = twin.standardise(fit)[0]
fit_linear = {}
for wd in (1e-4, 1e-2):
    w, b = twin.fit_linear(fit_std, fy, wd)
    fit_linear[str(wd)] = {"weight": packed(w), "bias": packed(b)}

f64 = [t.numpy() for t in twin.standardise(fit.double(), sel.double(), te.double())]
n_fit = len(fy)
logistic = []
for wd in twin.PROBE_WDS:
    model = LogisticRegression(C=1.0 / (n_fit * wd), l1_ratio=0.0, tol=1e-10,
                               max_iter=10000)
    model.fit(f64[0], fy.numpy())
    logistic.append({"weight_decay": wd,
                     "coef": packed(torch.from_numpy(model.coef_.T.copy())),
                     "select": packed(torch.from_numpy(model.predict(f64[1]))),
                     "test": packed(torch.from_numpy(model.predict(f64[2])))})

neighbours = KNeighborsClassifier(n_neighbors=twin.KNN_K, metric="cosine",
                                  algorithm="brute",
                                  weights=lambda d: np.exp((1 - d) / twin.KNN_T))
neighbours.fit(f64[0], fy.numpy())
knn = {name: packed(torch.from_numpy(neighbours.predict(x)))
       for name, x in (("select", f64[1]), ("test", f64[2]), ("loo", None))}



def minimised(params, objective, **settings):
    lbfgs = torch.optim.LBFGS(params, **settings)
    losses = []

    def closure():
        lbfgs.zero_grad()
        loss = objective()
        loss.backward()
        losses.append(loss.item())
        return loss

    lbfgs.step(closure)
    return losses


def rosenbrock(start, dtype):
    x = torch.tensor(start, dtype=dtype, requires_grad=True)
    losses = minimised([x], lambda: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
                       lr=1, max_iter=100, history_size=20,
                       line_search_fn="strong_wolfe")
    return {"losses": losses, "x": x.detach().tolist()}


SPD_A = [[26, -6, 0, -17, -12, -9], [-6, 29, -17, 11, 1, -14],
         [0, -17, 25, -3, 3, 14], [-17, 11, -3, 35, 8, 2],
         [-12, 1, 3, 8, 29, -1], [-9, -14, 14, 2, -1, 32]]
SPD_B = [3, -1, 4, -1, 5, -9]


def quadratic(line_search):
    a = torch.tensor(SPD_A, dtype=torch.float64)
    b = torch.tensor(SPD_B, dtype=torch.float64)
    x = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    minimised([x], lambda: 0.5 * (x * (a @ x)).sum() - (b * x).sum(),
              max_iter=100, tolerance_change=1e-12, line_search_fn=line_search)
    exact = torch.linalg.solve(a, b)
    return (x.detach() - exact).abs().max().item()


def probe_trace(n, scale, wd):
    x, y = scale * fit_std[:n], fy[:n]
    w = torch.zeros(24, 10, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    return minimised([w, b],
                     lambda: F.cross_entropy(x @ w + b, y) + 0.5 * wd * (w * w).sum(),
                     lr=1.0, max_iter=twin.PROBE_ITERS, tolerance_grad=1e-7,
                     tolerance_change=1e-10, history_size=20,
                     line_search_fn="strong_wolfe")


out = {"fit": packed(fit), "fit_labels": packed(fy),
       "select": packed(sel), "select_labels": packed(sy),
       "test": packed(te), "test_labels": packed(ty),
       "labels256": packed(labels256),
       "split": [packed(i) for i in split256],
       "full_split": full_split,
       "std": [packed(t) for t in twin.standardise(fit, sel, te)],
       "fit_linear": fit_linear,
       "suite": twin.probe_suite(fit, sel, te, fy, sy, ty),
       "sklearn": {"logistic": logistic, "knn": knn,
                   "version": sklearn.__version__},
       "lbfgs_rosenbrock": rosenbrock([-1.5, 2.0], torch.float64),
       "lbfgs_rosenbrock32": [{"start": start, **rosenbrock(start, torch.float32)}
                              for start in ([1.3, 2.0], [-1.2, 2.0])],
       "lbfgs_quadratic": {"none": quadratic(None),
                           "strong_wolfe": quadratic("strong_wolfe")},
       "lbfgs_probe": {"n": 100, "scale": 10.0, "weight_decay": 1e-2,
                       "losses": probe_trace(100, 10.0, 1e-2)}}
print(json.dumps(out))
