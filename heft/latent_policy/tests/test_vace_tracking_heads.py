import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/search_vace_tracking_heads.py"
spec = importlib.util.spec_from_file_location("vace_tracking", SCRIPT)
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


class TrackingDecodeTest(unittest.TestCase):
    def test_tracks_translations_without_future_ground_truth(self):
        q = torch.zeros(2, 3, 4, 4)
        q[:, 0, 0, 0] = 1
        q[:, 1, 1, 0] = 1
        q[:, 2, 2, 0] = 1
        q[:, 1, 2, 1] = 1
        q[:, 2, 3, 1] = 1
        anchor = np.array([0, 1])
        cells = np.array([[0, 3, 3], [3, 2, 0]])
        predictions = tracking.decode_matches(q, q, anchor, cells, 2)
        for predicted in predictions:
            np.testing.assert_array_equal(predicted[:, 0], [[0, 1, 2], [0, 1, 2]])
            np.testing.assert_array_equal(predicted[:, 1, 1:], [[2, 3], [2, 3]])
        altered = cells.copy()
        altered[0, 1:] = 0
        altered[1, [0, 2]] = 1
        for before, after in zip(predictions, tracking.decode_matches(q, q, anchor, altered, 2)):
            np.testing.assert_array_equal(before, after)

    def test_static_periods_do_not_inflate_displaced_score(self):
        gt = np.array([[[0, 0], [1, 0], [2, 0], [16, 0]]], dtype=np.float32)
        points = {"xy": gt, "groups": np.array(["moving_object"]),
                  "valid": np.array([[False, True, True, True]]),
                  "displaced": np.array([[False, False, False, True]])}
        static = np.zeros_like(gt)
        self.assertAlmostEqual(tracking.track_metrics(static, points, "moving_object")["within8"], 2 / 3)
        self.assertEqual(tracking.track_metrics(static, points, "moving_object", True)["within8"], 0)

    def test_pixel_centers_round_trip(self):
        cells = np.arange(32 * 32)
        xy = tracking.cell_to_xy(cells, 32, 256)
        np.testing.assert_array_equal(tracking.xy_to_cell(xy, 32, 256), cells)


if __name__ == "__main__":
    unittest.main()
