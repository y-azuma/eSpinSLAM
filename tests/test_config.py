from pathlib import Path

import pytest

from espinslam.config import load_config


def test_example_config_is_valid():
    config = load_config(Path("examples/configs/debris_davis346.jsonc"))
    assert config.data.evaluation_split == "test"
    assert config.frontend.chunk_events == 1000
    assert config.frequency.min_hz < config.frequency.max_hz


def test_non_test_evaluation_split_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"data":{"evaluation_split":"train"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="remain 'test'"):
        load_config(path)

