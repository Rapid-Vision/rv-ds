from pathlib import Path

import pytest

from rv_ds.errors import ValidationFailure
from rv_ds.plugin_loader import load_exporter, load_extractor


def test_load_builtin_plugins() -> None:
    extractor, ext_info = load_extractor(
        "default-seg",
        {
            "class_mapping": [
                {"class": "sphere", "required_tags": ["sphere"]},
            ]
        },
    )
    exporter, exp_info = load_exporter("default-yolo-seg", {})

    assert hasattr(extractor, "extract_dataset")
    assert hasattr(exporter, "export_dataset")
    assert ext_info.source == "builtin"
    assert exp_info.source == "builtin"


def test_load_path_extractor_plugin(tmp_path: Path) -> None:
    plugin = tmp_path / "extractor.py"
    plugin.write_text(
        """
from rv_ds.plugin_api import PluginOptions
from rv_ds.plugin_api import BaseExtractor


class ExtractorOptions(PluginOptions):
    dataset: object


class ExtractorPlugin(BaseExtractor):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({"instance_class"})

    def __init__(self, opts: ExtractorOptions):
        super().__init__(opts)
        self.opts = opts

    def extract_dataset(self, ctx):
        return self.opts.dataset
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


def test_invalid_options_model_raises(tmp_path: Path) -> None:
    plugin = tmp_path / "extractor_bad_options.py"
    plugin.write_text(
        """
from rv_ds.plugin_api import BaseExtractor


class ExtractorPlugin(BaseExtractor):
    OptionsModel = object
    produced_features = frozenset({"instance_class"})

    def extract_dataset(self, ctx):
        return None
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationFailure):
        load_extractor(str(plugin), {})


def test_missing_plugin_class_raises(tmp_path: Path) -> None:
    plugin = tmp_path / "extractor_no_class.py"
    plugin.write_text(
        """
from rv_ds.plugin_api import PluginOptions


class ExtractorOptions(PluginOptions):
    x: int = 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationFailure):
        load_extractor(str(plugin), {})
