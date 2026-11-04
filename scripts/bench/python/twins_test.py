import unittest

from pipelines import peak_rss_mb


class PeakRss(unittest.TestCase):
    def test_reads_ru_maxrss_in_the_platforms_unit(self):
        self.assertEqual(peak_rss_mb(3 * 2**30, "darwin"), 3072)
        self.assertEqual(peak_rss_mb(3 * 2**20, "linux"), 3072)


if __name__ == "__main__":
    unittest.main()
