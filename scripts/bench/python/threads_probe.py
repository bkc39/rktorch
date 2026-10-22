"""Fixture for the twin test of threads.py: scaling_run with a mix slower
than the run, with mixes that stamp their first call and with a mix that
raises, the last two also in a run that ends before any worker is
scheduled; and the main loop's zero_grad."""

import time

from common import emit
from threads import scaling_run, training_step

import torch


def slow_mix():
    def run():
        time.sleep(0.5)

    return run


def stamped(seconds):
    stamps = []

    def stamping():
        first = []
        stamps.append(first)

        def run():
            if not first:
                first.append(time.perf_counter())

        return run

    run = scaling_run(4, "idle", seconds, make_mix=stamping)
    return {"workers": len(stamps), "ran": sum(1 for first in stamps if first),
            "before_start": sum(1 for first in stamps
                                if first and first[0] < run["started_s"])}


def raised(seconds):
    made = []

    def failing_first():
        made.append(None)

        def fail():
            raise RuntimeError("unsupported")

        return fail if len(made) == 1 else (lambda: None)

    try:
        scaling_run(2, "idle", seconds, make_mix=failing_first)
    except RuntimeError as e:
        return str(e)
    return None


def zero_grad_modes():
    modes = []
    zero_grad = torch.optim.SGD.zero_grad

    def recording(self, set_to_none=True):
        modes.append(set_to_none)
        return zero_grad(self, set_to_none=set_to_none)

    torch.optim.SGD.zero_grad = recording
    try:
        step = training_step()
        step()
        step()
    finally:
        torch.optim.SGD.zero_grad = zero_grad
    return modes


def main():
    slow = scaling_run(1, "idle", 0.1, make_mix=slow_mix)
    emit({"case": "slow-mix", "ops_per_s": slow["ops_per_s"],
          "stopped": slow["stopped"]})
    emit({"case": "stamped", "runs": [stamped(s) for s in (0, 0.1)]})
    emit({"case": "failing", "raised": [raised(s) for s in (0, 0.1)]})
    emit({"case": "zero-grad", "set_to_none": zero_grad_modes()})


if __name__ == "__main__":
    main()
