from pathlib import Path

from medical_parsing.config import ASSET_FILENAMES, AssetBundle, load_config


def test_config_and_external_asset_names():
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert load_config() == config
    assert config.model.name == "google/medgemma-1.5-4b-it"
    assert config.model.vocab_chunk_size == 16
    assert config.model.token_chunk == 16
    assert config.model.scoring_row_batch_size == 4
    assert config.multilabel.scoring_row_batch_size == 4
    assert config.regression.retrieval_neighbors == 15
    assert config.regression.quantiles == (0.25, 0.5, 0.75)
    bundle = AssetBundle(Path("external-checkpoints-that-do-not-exist"))
    assert bundle.path("classification_heads").name == "classification_heads.pt"
    assert len(bundle.missing()) == len(ASSET_FILENAMES)


def test_legacy_streaming_config_key_is_accepted(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "model:\n  token_chunk: 7\n  scoring_row_batch_size: 3\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.vocab_chunk_size == 7
    assert config.multilabel.scoring_row_batch_size == 3
