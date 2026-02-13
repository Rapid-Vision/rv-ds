import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from .contracts import (
    INSTANCE_BBOX,
    INSTANCE_CLASS,
    INSTANCE_SEGMENT,
    SAMPLE_CLASS,
    STANDARD_FEATURES,
)
from .errors import ValidationFailure
from .ir import DatasetIR
from .scanner import SamplePaths


class PluginOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


TOptions = TypeVar("TOptions", bound=PluginOptions)


@dataclass
class ExtractionContext:
    dataset_dir: Path
    image_file: str
    samples: list[SamplePaths]
    framework_options: dict[str, Any]
    _warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self._warnings.append(message)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)


@dataclass
class ExportContext:
    dataset: DatasetIR
    output_dir: Path
    framework_options: dict[str, Any]
    _warnings: list[str] = field(default_factory=list)
    _outputs: set[Path] = field(default_factory=set)

    def warn(self, message: str) -> None:
        self._warnings.append(message)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def outputs(self) -> tuple[Path, ...]:
        return tuple(sorted(self._outputs))

    def safe_path(self, path: Path) -> Path:
        candidate = (
            (self.output_dir / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )
        root = self.output_dir.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValidationFailure(
                f"exporter attempted to use path outside output directory: '{candidate}'"
            )
        return candidate

    def mkdir(self, path: Path) -> Path:
        safe = self.safe_path(path)
        safe.mkdir(parents=True, exist_ok=True)
        return safe

    def write_text(self, path: Path, content: str) -> Path:
        safe = self.safe_path(path)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        self.add_output(safe)
        return safe

    def copy_image(self, src: Path, dst: Path) -> Path:
        safe = self.safe_path(dst)
        safe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, safe)
        self.add_output(safe)
        return safe

    def add_output(self, path: Path) -> Path:
        safe = self.safe_path(path)
        self._outputs.add(safe)
        return safe


@dataclass
class ExporterRunResult:
    stats: dict[str, int] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class BaseExtractor(Generic[TOptions], ABC):
    OptionsModel: ClassVar[type[PluginOptions]] = PluginOptions
    produced_features: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, opts: TOptions) -> None:
        self.opts = opts

    @abstractmethod
    def extract_dataset(self, ctx: ExtractionContext) -> DatasetIR:
        raise NotImplementedError


class BaseExporter(Generic[TOptions], ABC):
    OptionsModel: ClassVar[type[PluginOptions]] = PluginOptions
    required_features: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, opts: TOptions) -> None:
        self.opts = opts

    @abstractmethod
    def export_dataset(
        self, ctx: ExportContext
    ) -> ExporterRunResult | dict[str, Any]:
        raise NotImplementedError


__all__ = [
    "BaseExporter",
    "BaseExtractor",
    "ExportContext",
    "ExporterRunResult",
    "ExtractionContext",
    "INSTANCE_BBOX",
    "INSTANCE_CLASS",
    "INSTANCE_SEGMENT",
    "PluginOptions",
    "SAMPLE_CLASS",
    "STANDARD_FEATURES",
]
