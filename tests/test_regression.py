from PIL import Image

import numpy as np

from medical_parsing.tasks.regression import final_regression_blend, geometry_one, tile_views


def test_regression_geometry_and_views(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (300, 100), "gray").save(path)
    assert geometry_one(str(path)).shape == (960,)
    assert len(tile_views(Image.open(path))) == 4
    np.testing.assert_allclose(final_regression_blend(np.array([80.0]), np.array([4.0]), np.array([60.0])), np.array([76.5]))
