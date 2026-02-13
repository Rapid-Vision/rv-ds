from pathlib import Path

import pytest

from rv_ds.errors import ValidationFailure
from rv_ds.plugin_loader import load_exporter, load_extractor


def test_load_builtin_plugins() -> None:
    extractor, ext_info = load_extractor(
        "default-segment",
        {
            "class_mapping": [
                {"class": "sphere", "required_tags": ["sphere"]},
            ]
        },
    )
    exporter, exp_info = load_exporter("default-yolo", {})

    assert hasattr(extractor, "extract_dataset")
    assert hasattr(exporter, "export_dataset")
    assert ext_info.source == "builtin"
    assert exp_info.source == "builtin"


def test_load_path_extractor_plugin(tmp_path: Path) -> None:
    plugin = tmp_path / "extractor.py"
    plugin.write_text(
        """
from rv_ds.plugin_api import PluginOptions


class ExtractorOptions(PluginOptions):
    dataset: object


def build_extractor(opts: ExtractorOptions):
    class P:
        def extract_dataset(self, ctx):
            return opts.dataset
    return P()
""",
        encoding="utf-8",
    )

    extractor, info = load_extractor(str(plugin), {"dataset": object()})

    assert hasattr(extractor, "extract_dataset")
    assert info.source == "path"


def test_missing_builder_raises(tmp_path: Path) -> None:
    plugin = tmp_path / "bad.py"
    plugin.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ValidationFailure):
        load_exporter(str(plugin), {})


def test_missing_options_model_raises(tmp_path: Path) -> None:
    plugin = tmp_path / "extractor_no_model.py"
    plugin.write_text(
        """
def build_extractor(opts):
    class P:
        def extract_dataset(self, ctx):
            return None
    return P()
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationFailure):
        load_extractor(str(plugin), {})
