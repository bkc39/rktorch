"""RNG state on RKTORCH_PARITY_DEVICE, for one RKTORCH_RNG_STREAM.

The twin of torch/tests/rng-state-parity-test.rkt (P1-P3). The stream is a
torch.Generator(device) ("generator") or the device's default generator
("default"). Without RKTORCH_RNG_STATE it seeds 7, warms up with rand(5) and
randn(3), and prints the state and the draws that follow; with it (hex) it
sets that state and prints the draws that follow.
"""
import json
import os

import torch

device = os.environ.get("RKTORCH_PARITY_DEVICE", "cpu")
stream = os.environ.get("RKTORCH_RNG_STREAM", "generator")
state_hex = os.environ.get("RKTORCH_RNG_STATE")


def flat(t):
    return [float(v) for v in t.cpu().flatten().tolist()]


def after(g):
    return {
        "rand": flat(torch.rand(5, device=device, generator=g)),
        "randn": flat(torch.randn(3, 2, device=device, generator=g)),
    }


def default_state():
    if device == "cpu":
        return torch.get_rng_state()
    return torch.cuda.get_rng_state(0)


def set_default_state(state):
    if device == "cpu":
        torch.set_rng_state(state)
    else:
        torch.cuda.set_rng_state(state, 0)


if state_hex is None:
    if stream == "generator":
        g = torch.Generator(device).manual_seed(7)
    else:
        torch.manual_seed(7)
        g = None
    torch.rand(5, device=device, generator=g)
    torch.randn(3, device=device, generator=g)
    state = g.get_state() if g is not None else default_state()
    print(json.dumps({
        "state": {"hex": state.numpy().tobytes().hex(),
                  "shape": list(state.shape)},
        "after": after(g),
    }))
else:
    state = torch.frombuffer(bytearray(bytes.fromhex(state_hex)),
                             dtype=torch.uint8)
    if stream == "generator":
        g = torch.Generator(device)
        g.set_state(state)
    else:
        g = None
        set_default_state(state)
    print(json.dumps({"after": after(g)}))
