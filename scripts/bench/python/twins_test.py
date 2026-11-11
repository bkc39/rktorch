import os
import time
import unittest

from op_cost import stack_inputs, thread_loop
from pipelines import PIPELINES, JpegCycle, jpeg_files, peak_rss_mb

HYMENOPTERA_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                   "torch", "vision", "fixtures", "hymenoptera")


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


class JpegStages(unittest.TestCase):
    def test_decode_alone_yields_a_pixel_and_transforms_alone_an_image(self):
        files = jpeg_files(HYMENOPTERA_FIXTURE)
        shapes = [tuple(JpegCycle(8, stage, files)[5][0].shape)
                  for stage in ("both", "decode", "transforms")]
        self.assertEqual(shapes, [(3, 224, 224), (), (3, 224, 224)])

    def test_the_stages_run_by_the_racket_names(self):
        self.assertLessEqual({"jpeg", "jpeg-decode", "jpeg-transforms"}, set(PIPELINES))


if __name__ == "__main__":
    unittest.main()
