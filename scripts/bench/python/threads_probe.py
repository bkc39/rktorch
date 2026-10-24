"""Fixture for the twin test of threads.py: scaling_run with a mix slower
than the run, with mixes that stamp their first call and with a mix that
raises, the last two also in a run that ends before any worker is
scheduled; and the main loop's training step."""

import time

from common import emit
from threads import scaling_run, training_step


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


def trained_steps():
    step = training_step()
    steps = 0
    for _ in range(2):
        step()
        steps += 1
    return steps


def main():
    slow = scaling_run(1, "idle", 0.1, make_mix=slow_mix)
    emit({"case": "slow-mix", "ops_per_s": slow["ops_per_s"],
          "stopped": slow["stopped"]})
    emit({"case": "stamped", "runs": [stamped(s) for s in (0, 0.1)]})
    emit({"case": "failing", "raised": [raised(s) for s in (0, 0.1)]})
    emit({"case": "training-step", "steps": trained_steps()})


if __name__ == "__main__":
    main()
