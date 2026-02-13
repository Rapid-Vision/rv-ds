import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import ValidationFailure
from .ir import DatasetIR, SampleRecord
from .plugin_api import (
    BaseExporter,
    BaseExtractor,
    ExportContext,
    ExporterRunResult,
    ExtractionContext,
    ExtractorDatasetInfo,
    StreamRequest,
)
from .plugin_loader import load_exporter, load_extractor
from .scanner import SamplePaths, discover_samples

IteratorFactory = Callable[[StreamRequest, int | None], Iterator[SampleRecord]]


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


@dataclass
class PreparedContexts:
    samples: list[SamplePaths]
    export_dir: Path
    extractor: BaseExtractor[Any]
    extractor_info: Any
    exporter: BaseExporter[Any]
    exporter_info: Any
    extraction_ctx: ExtractionContext
    framework_opts_exporter: dict[str, Any]
    random_seed: int | None


@dataclass
class DescribePhaseResult:
    dataset_info: ExtractorDatasetInfo


@dataclass
class StreamState:
    processed_candidates: int = 0
    yielded_samples: int = 0
    stream_started: bool = False
    collected_samples: list[SampleRecord] = field(default_factory=list)


@dataclass
class ExportPhaseResult:
    export_ctx: ExportContext
    export_result: ExporterRunResult
    exporter_info: Any
    stream_state: StreamState


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
    prepared = prepare_contexts(config)
    describe_phase = run_extractor_describe(config, prepared)
    if prepared.extraction_ctx.warnings and config.fail_on_plugin_warning:
        joined = " | ".join(prepared.extraction_ctx.warnings)
        raise ValidationFailure(
            f"extractor produced warnings and fail-on-warning is set: {joined}"
        )
    export_phase = run_exporter(config, prepared, describe_phase)
    return finalize_meta(config, prepared, describe_phase, export_phase)


def prepare_contexts(config: ExportConfig) -> PreparedContexts:
    samples = discover_samples(config.dataset_dir, config.image_file)
    export_dir = _build_export_dir(config.output_dir)
    export_dir.mkdir(parents=True, exist_ok=False)

    framework_opts_extractor = _read_framework_opts(config.extractor_opts)
    framework_opts_exporter = _read_framework_opts(config.exporter_opts)
    random_seed = _read_framework_random_seed(
        framework_opts_extractor, framework_opts_exporter
    )

    extractor_opts = _strip_framework_opts(config.extractor_opts)
    exporter_opts = _strip_framework_opts(config.exporter_opts)

    extractor, extractor_info = load_extractor(config.extractor_spec, extractor_opts)
    exporter, exporter_info = load_exporter(config.exporter_spec, exporter_opts)
    _validate_feature_contract(
        extractor=extractor,
        exporter=exporter,
        extractor_spec=config.extractor_spec,
        exporter_spec=config.exporter_spec,
    )
    extraction_ctx = ExtractionContext(
        dataset_dir=config.dataset_dir,
        image_file=config.image_file,
        samples=samples,
        framework_options=framework_opts_extractor,
    )

    return PreparedContexts(
        samples=samples,
        export_dir=export_dir,
        extractor=extractor,
        extractor_info=extractor_info,
        exporter=exporter,
        exporter_info=exporter_info,
        extraction_ctx=extraction_ctx,
        framework_opts_exporter=framework_opts_exporter,
        random_seed=random_seed,
    )


def run_extractor_describe(
    config: ExportConfig, prepared: PreparedContexts
) -> DescribePhaseResult:
    try:
        dataset_info = prepared.extractor.describe_dataset(prepared.extraction_ctx)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailure(
            f"extractor '{config.extractor_spec}' failed while describing dataset: {exc}"
        ) from exc

    if not isinstance(dataset_info, ExtractorDatasetInfo):
        raise ValidationFailure(
            f"extractor '{config.extractor_spec}' returned unsupported dataset info type: "
            f"{type(dataset_info)!r}"
        )

    _validate_dataset_info(dataset_info)
    return DescribePhaseResult(dataset_info=dataset_info)


def run_exporter(
    config: ExportConfig,
    prepared: PreparedContexts,
    describe_phase: DescribePhaseResult,
) -> ExportPhaseResult:
    sample_iterator, stream_state = _build_sample_iterator(
        config=config,
        prepared=prepared,
        collect_samples=config.dump_ir,
    )
    export_ctx = ExportContext(
        dataset_info=describe_phase.dataset_info,
        _sample_iterator=sample_iterator,
        output_dir=prepared.export_dir,
        framework_options=prepared.framework_opts_exporter,
    )

    try:
        raw_export_result = prepared.exporter.export_dataset(export_ctx)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailure(
            f"exporter '{config.exporter_spec}' failed while exporting dataset: {exc}"
        ) from exc

    export_result = _normalize_exporter_result(raw_export_result)
    _validate_export_outputs(prepared.export_dir, export_ctx, export_result)

    if prepared.extraction_ctx.warnings and config.fail_on_plugin_warning:
        joined = " | ".join(prepared.extraction_ctx.warnings)
        raise ValidationFailure(
            f"extractor produced warnings and fail-on-warning is set: {joined}"
        )

    if export_ctx.warnings and config.fail_on_plugin_warning:
        joined = " | ".join(export_ctx.warnings)
        raise ValidationFailure(
            f"exporter produced warnings and fail-on-warning is set: {joined}"
        )

    if config.dump_ir:
        dataset = DatasetIR(
            samples=stream_state.collected_samples,
            class_names=describe_phase.dataset_info.class_names,
            meta=dict(describe_phase.dataset_info.meta),
        )
        dataset.validate()
        (prepared.export_dir / "ir_dump.json").write_text(
            json.dumps(dataset.to_dict(), indent=2), encoding="utf-8"
        )

    return ExportPhaseResult(
        export_ctx=export_ctx,
        export_result=export_result,
        exporter_info=prepared.exporter_info,
        stream_state=stream_state,
    )


def _build_sample_iterator(
    config: ExportConfig,
    prepared: PreparedContexts,
    collect_samples: bool,
) -> tuple[IteratorFactory, StreamState]:
    state = StreamState()

    def _iter(request: StreamRequest, max_samples: int | None) -> Iterator[SampleRecord]:
        if state.stream_started:
            raise ValidationFailure(
                "exporter requested sample stream multiple times; only one pass is supported"
            )

        state.stream_started = True
        sample_paths = list(prepared.samples)
        if request.order == "random":
            rng = random.Random(prepared.random_seed)
            rng.shuffle(sample_paths)

        yielded_this_call = 0
        for sample_path in sample_paths:
            state.processed_candidates += 1
            try:
                extracted = prepared.extractor.extract_sample(
                    prepared.extraction_ctx, sample_path
                )
            except Exception as exc:  # noqa: BLE001
                raise ValidationFailure(
                    "extractor "
                    f"'{config.extractor_spec}' failed while processing sample "
                    f"'{sample_path.sample_id}': {exc}"
                ) from exc

            if extracted is None:
                continue

            extracted.validate()
            state.yielded_samples += 1
            if collect_samples:
                state.collected_samples.append(extracted)

            yield extracted

            yielded_this_call += 1
            if max_samples is not None and yielded_this_call >= max_samples:
                break

    return _iter, state


def _validate_feature_contract(
    extractor: BaseExtractor[Any],
    exporter: BaseExporter[Any],
    extractor_spec: str,
    exporter_spec: str,
) -> None:
    produced = set(extractor.produced_features)
    required = set(exporter.required_features)
    missing = required - produced
    if missing:
        raise ValidationFailure(
            "extractor/exporter feature contract mismatch:\n"
            f"extractor='{extractor_spec}'\n\tproduced={sorted(produced)};\n"
            f"exporter='{exporter_spec}'\n\trequired={sorted(required)};\n"
            f"missing={sorted(missing)}"
        )


def _validate_dataset_info(dataset_info: ExtractorDatasetInfo) -> None:
    if not all(isinstance(name, str) for name in dataset_info.class_names):
        raise ValidationFailure("extractor dataset info class_names must be list[str]")
    if len(set(dataset_info.class_names)) != len(dataset_info.class_names):
        raise ValidationFailure("extractor dataset info class_names must be unique")
    if not isinstance(dataset_info.meta, dict):
        raise ValidationFailure("extractor dataset info meta must be object")


def finalize_meta(
    config: ExportConfig,
    prepared: PreparedContexts,
    describe_phase: DescribePhaseResult,
    export_phase: ExportPhaseResult,
) -> ExportResult:
    _ = describe_phase
    stats = ExportStats(
        discovered_samples=len(prepared.samples),
        extracted_samples=export_phase.stream_state.yielded_samples,
        exporter_outputs=(
            len(export_phase.export_result.outputs)
            if export_phase.export_result.outputs
            else len(export_phase.export_ctx.outputs)
        ),
    )

    _write_meta(
        path=prepared.export_dir / "rv_ds_meta.json",
        config=config,
        stats=stats,
        extractor_info=prepared.extractor_info,
        exporter_info=export_phase.exporter_info,
        extraction_warnings=list(prepared.extraction_ctx.warnings),
        export_warnings=list(export_phase.export_ctx.warnings),
        exporter_result=export_phase.export_result,
        stream_state=export_phase.stream_state,
    )

    return ExportResult(export_dir=prepared.export_dir, stats=stats)


def _read_framework_opts(opts: dict[str, Any]) -> dict[str, Any]:
    raw = opts.get("_framework", {})
    if not isinstance(raw, dict):
        raise ValidationFailure("reserved options key '_framework' must be an object")
    return raw


def _read_framework_random_seed(
    extractor_framework_opts: dict[str, Any],
    exporter_framework_opts: dict[str, Any],
) -> int | None:
    extractor_seed_raw = extractor_framework_opts.get("random_seed")
    exporter_seed_raw = exporter_framework_opts.get("random_seed")
    extractor_seed: int | None = None
    exporter_seed: int | None = None

    for source, value in (
        ("extractor", extractor_seed_raw),
        ("exporter", exporter_seed_raw),
    ):
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationFailure(
                f"_framework.random_seed for {source} must be integer"
            )
        if source == "extractor":
            extractor_seed = value
        else:
            exporter_seed = value

    if (
        extractor_seed is not None
        and exporter_seed is not None
        and extractor_seed != exporter_seed
    ):
        raise ValidationFailure(
            "_framework.random_seed mismatch between extractor and exporter options"
        )

    if extractor_seed is not None:
        return extractor_seed
    return exporter_seed


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
        output_path = Path(output_str)
        resolved_output = (
            output_path.resolve()
            if output_path.is_absolute()
            else (root / output_path).resolve()
        )
        if resolved_output != root and root not in resolved_output.parents:
            raise ValidationFailure(
                f"exporter reported output outside export directory: '{resolved_output}'"
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
    stream_state: StreamState,
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
        "stream": {
            "processed_candidates": stream_state.processed_candidates,
            "yielded_samples": stream_state.yielded_samples,
        },
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
