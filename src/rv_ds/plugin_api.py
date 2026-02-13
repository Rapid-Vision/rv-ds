import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from .errors import ValidationFailure
from .ir import DatasetIR
from .scanner import SamplePaths


class PluginOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@dataclass
class ExtractionContext:
    dataset_dir: Path
    image_file: str
    samples: list[SamplePaths]
    framework_options: dict[str, Any]
    _warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self._warnings.append(message)


@dataclass
class ExportContext:
    dataset: DatasetIR
    output_dir: Path
    framework_options: dict[str, Any]
    _warnings: list[str] = field(default_factory=list)
    _outputs: set[Path] = field(default_factory=set)

    def warn(self, message: str) -> None:
        self._warnings.append(message)

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
        self._outputs.add(safe)
        return safe

    def copy_image(self, src: Path, dst: Path) -> Path:
        safe = self.safe_path(dst)
        safe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, safe)
        self._outputs.add(safe)
        return safe


@dataclass
class ExporterRunResult:
    stats: dict[str, int] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class Extractor(Protocol):
    def extract_dataset(self, ctx: ExtractionContext) -> DatasetIR: ...


class Exporter(Protocol):
    def export_dataset(
        self, ctx: ExportContext
    ) -> ExporterRunResult | dict[str, Any]: ...
