from pathlib import Path

from medical_parsing.config import ASSET_FILENAMES, AssetBundle, load_config


def test_config_and_external_asset_names():
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert config.model.name == "google/medgemma-1.5-4b-it"
    assert config.model.token_chunk == 16
    assert config.model.scoring_row_batch_size == 4
    bundle = AssetBundle(Path("external-checkpoints-that-do-not-exist"))
    assert bundle.path("classification_heads").name == "classification_heads.pt"
    assert len(bundle.missing()) == len(ASSET_FILENAMES)
