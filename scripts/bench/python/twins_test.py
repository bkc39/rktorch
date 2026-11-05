import time
import unittest

from op_cost import stack_inputs, thread_loop
from pipelines import peak_rss_mb


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


if __name__ == "__main__":
    unittest.main()
