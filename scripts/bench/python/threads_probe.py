"""Fixture for the twin test of threads.py: scaling_run with a mix slower
than the run, with mixes that stamp their first call, and with a mix that
raises."""

import time

from common import emit
from threads import scaling_run


def slow_mix():
    def run():
        time.sleep(0.5)

    return run


def main():
    slow = scaling_run(1, "idle", 0.1, make_mix=slow_mix)
    stamps = []

    def stamping():
        first = []
        stamps.append(first)

        def run():
            if not first:
                first.append(time.perf_counter())

        return run

    stamped = scaling_run(4, "idle", 0.1, make_mix=stamping)
    emit({"case": "slow-mix", "ops_per_s": slow["ops_per_s"],
          "stopped": slow["stopped"]})
    emit({"case": "stamped", "workers": len(stamps),
          "before_start": sum(1 for first in stamps
                              if first and first[0] < stamped["started_s"])})
    made = []

    def failing_first():
        made.append(None)

        def fail():
            raise RuntimeError("unsupported")

        return fail if len(made) == 1 else (lambda: None)

    try:
        scaling_run(2, "idle", 0.1, make_mix=failing_first)
        raised = None
    except RuntimeError as e:
        raised = str(e)
    emit({"case": "failing", "raised": raised})


if __name__ == "__main__":
    main()
