import time
import unittest

from op_cost import thread_loop
from pipelines import peak_rss_mb


class ThreadLoop(unittest.TestCase):
    def test_counts_only_the_steps_the_window_finished(self):
        step_s = 0.1
        ops, elapsed = thread_loop(1, lambda: time.sleep(step_s), 0.25)
        self.assertGreaterEqual(ops, 1)
        self.assertLessEqual(ops * step_s, elapsed, (ops, elapsed))


class PeakRss(unittest.TestCase):
    def test_reads_ru_maxrss_in_the_platforms_unit(self):
        self.assertEqual(peak_rss_mb(3 * 2**30, "darwin"), 3072)
        self.assertEqual(peak_rss_mb(3 * 2**20, "linux"), 3072)


if __name__ == "__main__":
    unittest.main()
