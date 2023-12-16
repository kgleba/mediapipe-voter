import unittest
import random
from sklearn.datasets import make_blobs
from server import determine_n_clusters


class TestClustering(unittest.TestCase):
    def test_n_clusters(self):
        n_max = 8

        for n in range(2, n_max + 1):
            dataset, _ = make_blobs(n_samples=random.randint(100, 1000), centers=n, cluster_std=random.random(), random_state=0)
            self.assertEqual(determine_n_clusters(dataset), n)


if __name__ == '__main__':
    unittest.main()
