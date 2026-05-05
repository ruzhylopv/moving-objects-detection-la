import unittest

import numpy as np

from svd import svd


class TestSVD(unittest.TestCase):
    def assert_svd_matches_numpy(self, matrix, tol=1e-10):
        np.random.seed(0)
        u_custom, s_custom, vt_custom = svd(matrix, tol=tol)
        u_numpy, s_numpy, vt_numpy = np.linalg.svd(matrix, full_matrices=False)

        expected_rank = np.count_nonzero(s_numpy > tol)

        self.assertEqual(u_custom.shape, (matrix.shape[0], expected_rank))
        self.assertEqual(s_custom.shape, (expected_rank,))
        self.assertEqual(vt_custom.shape, (expected_rank, matrix.shape[1]))

        np.testing.assert_allclose(s_custom, s_numpy[:expected_rank], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            u_custom @ np.diag(s_custom) @ vt_custom,
            matrix,
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            u_custom.T @ u_custom,
            np.eye(expected_rank),
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            vt_custom @ vt_custom.T,
            np.eye(expected_rank),
            rtol=0,
            atol=1e-8,
        )

        # Singular vectors are only unique up to sign, so compare overlaps.
        u_overlap = np.abs(u_custom.T @ u_numpy[:, :expected_rank])
        vt_overlap = np.abs(vt_custom @ vt_numpy[:expected_rank, :].T)

        np.testing.assert_allclose(u_overlap, np.eye(expected_rank), rtol=0, atol=1e-6)
        np.testing.assert_allclose(vt_overlap, np.eye(expected_rank), rtol=0, atol=1e-6)

    def test_matches_numpy_on_full_rank_matrices(self):
        matrices = {
            "tall": np.array(
                [
                    [3, -1, 4],
                    [2, 5, -2],
                    [7, 0, 1],
                    [-4, 6, 3],
                ],
                dtype=float,
            ),
            "square_4x4": np.array(
                [
                    [17, -4, 9, 2],
                    [6, 13, -7, 5],
                    [-3, 8, 11, -6],
                    [10, -2, 4, 15],
                ],
                dtype=float,
            ),
            "wide": np.array(
                [
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [7, 8, 10, 11],
                ],
                dtype=float,
            ),
        }

        for name, matrix in matrices.items():
            with self.subTest(matrix=name):
                self.assert_svd_matches_numpy(matrix)

    def test_matches_numpy_on_rank_deficient_matrix(self):
        matrix = np.array(
            [
                [1, 2, 3],
                [2, 4, 6],
                [3, 6, 9],
                [4, 8, 12],
            ],
            dtype=float,
        )

        self.assert_svd_matches_numpy(matrix)


if __name__ == "__main__":
    unittest.main()
