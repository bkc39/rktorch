import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from common import meta
from op_cost import stack_inputs, thread_loop
from pipelines import peak_rss_mb
from threads import MIXES, PeakSampler, os_thread_count, peak_of, scaling_run

import torch


class ThreadLoop(unittest.TestCase):
    def test_counts_only_the_steps_the_window_finished(self):
        step_s = 0.1
        ops, elapsed = thread_loop(1, lambda: lambda: time.sleep(step_s), 0.25)
        self.assertGreaterEqual(ops, 1)
        self.assertLessEqual(ops * step_s, elapsed, (ops, elapsed))

    def test_a_failed_worker_fails_the_sample_and_ends_the_window(self):
        calls = []

        def step():
            calls.append(None)
            if len(calls) == 5:
                raise MemoryError("worker step failed")
            time.sleep(0.001)

        t0 = time.perf_counter()
        with self.assertRaisesRegex(MemoryError, "worker step failed"):
            thread_loop(2, lambda: step, 30)
        self.assertLess(time.perf_counter() - t0, 10)

    def test_each_worker_runs_its_own_case_as_the_racket_workers_do(self):
        cases = []

        def make_step():
            calls = []
            cases.append(calls)
            return lambda: (calls.append(None), time.sleep(0.001))

        thread_loop(3, make_step, 0.1)
        self.assertEqual(len(cases), 3)
        self.assertTrue(all(cases), [len(c) for c in cases])


class StackInputs(unittest.TestCase):
    def test_stacks_32_distinct_tensors_as_the_racket_case_does(self):
        ts = stack_inputs()
        self.assertEqual(len(ts), 32)
        self.assertEqual(len({t.data_ptr() for t in ts}), 32)


class PeakRss(unittest.TestCase):
    def test_reads_ru_maxrss_in_the_platforms_unit(self):
        self.assertEqual(peak_rss_mb(3 * 2**30, "darwin"), 3072)
        self.assertEqual(peak_rss_mb(3 * 2**20, "linux"), 3072)


def touch(directory, names):
    for name in names:
        open(os.path.join(directory, name), "w").close()


ABSENT = "/nonexistent/task"


class PeakOsThreads(unittest.TestCase):
    def test_counts_the_task_directorys_entries_and_none_without_one(self):
        with tempfile.TemporaryDirectory() as tasks:
            touch(tasks, ["1", "2", "3"])
            self.assertEqual(os_thread_count(tasks), 3)
        self.assertIsNone(os_thread_count(ABSENT))
        if os.path.isdir("/proc/self/task"):
            self.assertGreater(os_thread_count(), 0)

    def test_the_sampler_keeps_the_most_entries_it_saw(self):
        with tempfile.TemporaryDirectory() as tasks:
            touch(tasks, ["1"])
            sampler = PeakSampler(tasks, interval=0.01)
            touch(tasks, ["2", "3", "4"])
            time.sleep(0.2)
            for name in ["2", "3", "4"]:
                os.remove(os.path.join(tasks, name))
            time.sleep(0.2)
            self.assertEqual(sampler.stop(), 4)
        self.assertIsNone(PeakSampler(ABSENT, interval=0.01).stop())
        self.assertIsNone(peak_of([3, None]))
        self.assertEqual(peak_of([3, 5, 4]), 5)

    def test_a_run_counts_a_thread_per_worker_over_the_tasks_before_it(self):
        run = scaling_run(2, "idle", 0.3)
        if os.path.isdir("/proc/self/task"):
            self.assertGreater(run["baseline_os_threads"], 0)
            self.assertGreaterEqual(run["peak_os_threads"], run["baseline_os_threads"] + 2, run)
        elsewhere = scaling_run(1, "idle", 0.1, tasks=ABSENT)
        self.assertIsNone(elsewhere["peak_os_threads"])
        self.assertIsNone(elsewhere["baseline_os_threads"])


def print_counts(mixes, rounds):
    runs = [[scaling_run(2, "idle", 0.5, *MIXES[mix]) for _ in range(int(rounds))]
            for mix in mixes.split(",")]
    print(json.dumps([[[r["baseline_os_threads"], r["peak_os_threads"]] for r in rs]
                      for rs in runs]))


def counts_under(omp, mixes, rounds):
    env = {**os.environ, "OMP_NUM_THREADS": str(omp)}
    env.pop("MKL_NUM_THREADS", None)
    out = subprocess.run(
        [sys.executable, "-c", "import sys, twins_test; twins_test.print_counts(*sys.argv[1:])",
         mixes, str(rounds)],
        cwd=os.path.dirname(os.path.abspath(__file__)), env=env, check=True,
        capture_output=True, text=True, timeout=300)
    return json.loads(out.stdout)


@unittest.skipUnless(os.path.isdir("/proc/self/task"), "no /proc/self/task")
class IntraOpTeams(unittest.TestCase):
    def test_a_runs_peak_counts_each_workers_intra_op_team(self):
        [[[_, one]]] = counts_under(1, "grain", 1)
        [[[_, four]]] = counts_under(4, "grain", 1)
        self.assertGreaterEqual(four, one + 3)

    def test_back_to_back_runs_start_from_the_same_threads_and_peak_alike(self):
        [[[base_1, peak_1], [base_2, peak_2]]] = counts_under(4, "grain", 2)
        self.assertLessEqual(abs(base_2 - base_1), 1)
        self.assertLessEqual(abs(peak_2 - peak_1), 1)


class Mixes(unittest.TestCase):
    def test_each_mix_counts_its_own_ops(self):
        self.assertEqual(sorted(MIXES), ["grain", "small"])
        for name, ops in [("small", 4), ("grain", 2)]:
            make_mix, per_mix = MIXES[name]
            self.assertEqual(per_mix, ops)
            run = scaling_run(1, "idle", 0.1, make_mix, per_mix)
            self.assertGreater(run["mixes"], 0, name)
            self.assertAlmostEqual(run["ops_per_s"] * run["elapsed_s"] / run["mixes"], ops,
                                   places=9, msg=name)


class Meta(unittest.TestCase):
    def test_reports_the_thread_settings_the_twin_ran_under(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "3"}):
            m = meta(torch.device("cpu"))
        self.assertEqual((m["omp_num_threads"], m["mkl_num_threads"]), ("2", "3"))
        with mock.patch.dict(os.environ, {}, clear=True):
            m = meta(torch.device("cpu"))
        self.assertEqual((m["omp_num_threads"], m["mkl_num_threads"]), (None, None))


if __name__ == "__main__":
    unittest.main()
