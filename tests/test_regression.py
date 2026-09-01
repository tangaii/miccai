from PIL import Image

import numpy as np

from medical_parsing.tasks.regression import (
    build_adaptive_image_views,
    final_regression_blend,
    geometry_one,
    retrieve_reg,
    weighted_median,
)


def test_regression_geometry_and_views(tmp_path):
    paths = {
        "wide": tmp_path / "wide.png",
        "tall": tmp_path / "tall.png",
        "square": tmp_path / "square.png",
    }
    sizes = {"wide": (300, 100), "tall": (100, 300), "square": (200, 200)}
    for name, path in paths.items():
        Image.new("RGB", sizes[name], "gray").save(path)
        expected_views = 5 if name == "square" else 4
        assert len(build_adaptive_image_views(Image.open(path))) == expected_views
    assert geometry_one(str(paths["wide"])).shape == (960,)
    np.testing.assert_allclose(final_regression_blend(np.array([80.0]), np.array([4.0]), np.array([60.0])), np.array([76.5]))


def test_regression_retrieval_excludes_query_group_and_is_deterministic():
    source_rep = np.eye(3, dtype=np.float64)
    groups = np.array(["query", "other-a", "other-b"])
    targets = np.array([99.0, 10.0, 20.0])
    result = retrieve_reg(np.array([1.0, 0.0, 0.0]), "query", source_rep, groups, targets, ["q", "a", "b"], neighbors=2)
    assert result == 10.0
    assert weighted_median(np.array([2.0, 1.0]), np.array([1.0, 1.0]), ["b", "a"]) == 1.0
