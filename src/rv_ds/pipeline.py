import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import ValidationFailure
from .ir import DatasetIR
from .plugin_api import ExportContext, ExporterRunResult, ExtractionContext
from .plugin_loader import load_exporter, load_extractor
from .scanner import discover_samples


@dataclass(frozen=True)
class ExportConfig:
    dataset_dir: Path
    output_dir: Path
    image_file: str
    extractor_spec: str
    extractor_opts: dict[str, Any]
    exporter_spec: str
    exporter_opts: dict[str, Any]
    fail_on_plugin_warning: bool
    dump_ir: bool


@dataclass
class ExportStats:
    discovered_samples: int = 0
    extracted_samples: int = 0
    exporter_outputs: int = 0


@dataclass
class ExportResult:
    export_dir: Path
    stats: ExportStats


def load_json_opts(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    if not path.exists() or not path.is_file():
        raise ValidationFailure(f"options file does not exist: '{path}'")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(
            f"failed to parse options JSON '{path}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValidationFailure(f"options JSON '{path}' must be an object")

    return payload


def run_export(config: ExportConfig) -> ExportResult:
    samples = discover_samples(config.dataset_dir, config.image_file)
    export_dir = _build_export_dir(config.output_dir)
    export_dir.mkdir(parents=True, exist_ok=False)

    framework_opts_extractor = _read_framework_opts(config.extractor_opts)
    framework_opts_exporter = _read_framework_opts(config.exporter_opts)
    extractor_opts = _strip_framework_opts(config.extractor_opts)
    exporter_opts = _strip_framework_opts(config.exporter_opts)

    extractor, extractor_info = load_extractor(config.extractor_spec, extractor_opts)
    extraction_ctx = ExtractionContext(
        dataset_dir=config.dataset_dir,
        image_file=config.image_file,
        samples=samples,
        framework_options=framework_opts_extractor,
    )

    try:
        dataset = extractor.extract_dataset(extraction_ctx)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailure(
            f"extractor '{config.extractor_spec}' failed while processing dataset: {exc}"
        ) from exc

    if not isinstance(dataset, DatasetIR):
        raise ValidationFailure(
            f"extractor '{config.extractor_spec}' returned unsupported result type: "
            f"{type(dataset)!r}"
        )

    dataset.validate()

    if extraction_ctx.warnings and config.fail_on_plugin_warning:
        joined = " | ".join(extraction_ctx.warnings)
        raise ValidationFailure(
            f"extractor produced warnings and fail-on-warning is set: {joined}"
        )

    if config.dump_ir:
        (export_dir / "ir_dump.json").write_text(
            json.dumps(dataset.to_dict(), indent=2), encoding="utf-8"
        )

    exporter, exporter_info = load_exporter(config.exporter_spec, exporter_opts)
    export_ctx = ExportContext(
        dataset=dataset,
        output_dir=export_dir,
        framework_options=framework_opts_exporter,
    )

    try:
        raw_export_result = exporter.export_dataset(export_ctx)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailure(
            f"exporter '{config.exporter_spec}' failed while exporting dataset: {exc}"
        ) from exc

    export_result = _normalize_exporter_result(raw_export_result)
    _validate_export_outputs(export_dir, export_ctx, export_result)

    if export_ctx.warnings and config.fail_on_plugin_warning:
        joined = " | ".join(export_ctx.warnings)
        raise ValidationFailure(
            f"exporter produced warnings and fail-on-warning is set: {joined}"
        )

    stats = ExportStats(
        discovered_samples=len(samples),
        extracted_samples=len(dataset.samples),
        exporter_outputs=(
            len(export_result.outputs)
            if export_result.outputs
            else len(export_ctx.outputs)
        ),
    )

    _write_meta(
        path=export_dir / "rv_ds_meta.json",
        config=config,
        stats=stats,
        extractor_info=extractor_info,
        exporter_info=exporter_info,
        extraction_warnings=list(extraction_ctx.warnings),
        export_warnings=list(export_ctx.warnings),
        exporter_result=export_result,
    )

    return ExportResult(export_dir=export_dir, stats=stats)


def _read_framework_opts(opts: dict[str, Any]) -> dict[str, Any]:
    raw = opts.get("_framework", {})
    if not isinstance(raw, dict):
        raise ValidationFailure("reserved options key '_framework' must be an object")
    return raw


def _strip_framework_opts(opts: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in opts.items() if key != "_framework"}


def _normalize_exporter_result(
    raw: ExporterRunResult | dict[str, Any],
) -> ExporterRunResult:
    if isinstance(raw, ExporterRunResult):
        return raw

    if not isinstance(raw, dict):
        raise ValidationFailure(
            f"exporter result must be ExporterRunResult or dict, got {type(raw)!r}"
        )

    stats_raw: Any
    outputs_raw: Any
    meta_raw: Any

    if "stats" in raw or "outputs" in raw or "meta" in raw:
        stats_raw = raw.get("stats", {})
        outputs_raw = raw.get("outputs", [])
        meta_raw = raw.get("meta", {})
    else:
        stats_raw = {key: value for key, value in raw.items() if isinstance(value, int)}
        meta_raw = {key: value for key, value in raw.items() if key not in stats_raw}
        outputs_raw = []

    if not isinstance(stats_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, int) for k, v in stats_raw.items()
    ):
        raise ValidationFailure("exporter result 'stats' must be object[str, int]")
    if not isinstance(outputs_raw, list) or not all(
        isinstance(item, str) for item in outputs_raw
    ):
        raise ValidationFailure("exporter result 'outputs' must be list[str]")
    if not isinstance(meta_raw, dict):
        raise ValidationFailure("exporter result 'meta' must be object")

    return ExporterRunResult(stats=stats_raw, outputs=outputs_raw, meta=meta_raw)


def _validate_export_outputs(
    export_dir: Path,
    ctx: ExportContext,
    export_result: ExporterRunResult,
) -> None:
    root = export_dir.resolve()

    for output_str in export_result.outputs:
        output_path = Path(output_str).resolve()
        if output_path != root and root not in output_path.parents:
            raise ValidationFailure(
                f"exporter reported output outside export directory: '{output_path}'"
            )

    for safe_path in ctx.outputs:
        resolved = safe_path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValidationFailure(
                f"exporter attempted to write outside export directory: '{resolved}'"
            )


def _build_export_dir(base_output: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = base_output / timestamp

    if not export_dir.exists():
        return export_dir

    suffix = 1
    while True:
        candidate = base_output / f"{timestamp}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _opts_checksum(opts: dict[str, Any]) -> str:
    payload = json.dumps(opts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_meta(
    path: Path,
    config: ExportConfig,
    stats: ExportStats,
    extractor_info: Any,
    exporter_info: Any,
    extraction_warnings: list[str],
    export_warnings: list[str],
    exporter_result: ExporterRunResult,
) -> None:
    payload = {
        "config": {
            **asdict(config),
            "dataset_dir": str(config.dataset_dir),
            "output_dir": str(config.output_dir),
            "extractor_opts_checksum": _opts_checksum(config.extractor_opts),
            "exporter_opts_checksum": _opts_checksum(config.exporter_opts),
        },
        "plugins": {
            "extractor": asdict(extractor_info),
            "exporter": asdict(exporter_info),
        },
        "stats": asdict(stats),
        "warnings": {
            "extractor": extraction_warnings,
            "exporter": export_warnings,
        },
        "exporter_result": {
            "stats": exporter_result.stats,
            "outputs": exporter_result.outputs,
            "meta": exporter_result.meta,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
