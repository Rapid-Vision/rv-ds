import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn, cast

from .app_config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_IMAGE_FILE,
    DEFAULT_OUTPUT_DIR,
    AppConfig,
    DatasetConfig,
    DebugConfig,
    PluginConfig,
    PipelineConfig,
    ResolvedAppConfig,
    build_pipeline_export_config,
    dump_app_config,
    list_builtin_capabilities,
    load_app_config,
)
from .errors import RVExportError, ValidationFailure
from .inspector import InspectReport, inspect_dataset
from .pipeline import run_export
from .plugin_api import ExtractionContext
from .plugin_loader import load_exporter, load_extractor
from .scanner import discover_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rv-ds", description="Dataset inspection and export CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Generate a YAML config through a terminal wizard"
    )
    init_parser.add_argument("dataset_dir", nargs="?", type=Path, default=Path("."))
    init_parser.add_argument(
        "--output-config",
        "-o",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help="Path to write the generated YAML config",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing config file"
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect dataset structure and tags"
    )
    inspect_parser.add_argument("dataset_dir", type=Path)
    inspect_parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )

    export_parser = subparsers.add_parser(
        "export", help="Export a dataset using a YAML config"
    )
    export_parser.add_argument("--config", "-c", required=True, type=Path)
    export_parser.add_argument(
        "--dry-run", action="store_true", help="Validate without exporting"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a YAML config and dataset"
    )
    validate_parser.add_argument("--config", "-c", required=True, type=Path)

    subparsers.add_parser("list", help="List supported tasks, outputs, and presets")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            return _handle_init(args.dataset_dir, args.output_config, args.force)
        if args.command == "inspect":
            return _handle_inspect(args.dataset_dir, args.json)
        if args.command == "export":
            return _handle_export(args.config, args.dry_run)
        if args.command == "validate":
            return _handle_validate(args.config)
        if args.command == "list":
            return _handle_list()
        parser.error("unsupported command")
    except RVExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _handle_init(dataset_dir: Path, output_config: Path, force: bool) -> int:
    dataset_dir = dataset_dir.resolve()
    output_config = _resolve_output_config_path(output_config)
    if output_config.exists() and not force:
        raise ValidationFailure(
            f"config file already exists: '{output_config}'. Use --force to overwrite it."
        )

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValidationFailure(f"dataset directory does not exist: '{dataset_dir}'")

    report = inspect_dataset(dataset_dir, DEFAULT_IMAGE_FILE)
    if report.total_sample_dirs == 0:
        raise ValidationFailure(
            f"dataset directory '{dataset_dir}' has no sample folders"
        )
    if report.valid_sample_count == 0:
        raise ValidationFailure(
            "cannot initialize config because the dataset has no fully valid samples"
        )

    print(f"Dataset: {dataset_dir}")
    print(f"Valid samples: {report.valid_sample_count}/{report.total_sample_dirs}")
    if report.object_tags:
        print("Discovered object tags:")
        for index, item in enumerate(report.object_tags, start=1):
            print(f"  {index}. {item.tag} ({item.count})")
    else:
        print("Discovered object tags: none")

    task = _prompt_choice(
        "Select task",
        [
            ("1", "detection"),
            ("2", "segmentation"),
            ("3", "both"),
        ],
        default="2",
    )
    output_format = _prompt_output_format(task)
    selected_tags = _prompt_tag_selection(report)
    if not selected_tags:
        raise ValidationFailure(
            "at least one class mapping is required to generate a config"
        )

    output_dir = _prompt_text("Output directory", DEFAULT_OUTPUT_DIR)
    include_empty_default = "y" if output_format == "preview" else "n"
    include_empty = _prompt_yes_no(
        "Include empty samples", default=include_empty_default
    )

    app_config = AppConfig(
        dataset=DatasetConfig(path=dataset_dir),
        pipeline=PipelineConfig(output_dir=Path(output_dir)),
        extractor=PluginConfig(
            spec=_select_builtin_extractor_spec(task),
            options={
                "class_mapping": [
                    {"class": tag, "required_tags": [tag]} for tag in selected_tags
                ],
                "include_empty": include_empty,
            },
        ),
        exporter=PluginConfig(
            spec=_select_builtin_exporter_spec(task, output_format),
            options=_build_builtin_exporter_options(output_format, include_empty),
        ),
        debug=DebugConfig(),
    )

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(dump_app_config(app_config), encoding="utf-8")

    print(f"Wrote config: {output_config}")
    print(f"Next: rv-ds export --config {output_config}")
    return 0


def _handle_inspect(dataset_dir: Path, as_json: bool) -> int:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValidationFailure(f"dataset directory does not exist: '{dataset_dir}'")

    report = inspect_dataset(dataset_dir, DEFAULT_IMAGE_FILE)
    if as_json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
        return 0

    _print_inspect_report(report)
    return 0


def _handle_export(config_path: Path, dry_run: bool) -> int:
    resolved = load_app_config(config_path)
    validation_summary = validate_app_config(resolved)
    if dry_run:
        print(f"config_ok: {resolved.config_path}")
        print(
            "summary: "
            f"valid_samples={validation_summary.valid_samples} "
            f"class_names={len(validation_summary.class_names)} "
            f"output_format={resolved.app.pipeline.output_format}"
        )
        return 0

    result = run_export(build_pipeline_export_config(resolved))
    print(f"export_dir={result.export_dir}")
    print(
        "summary: "
        f"discovered_samples={result.stats.discovered_samples} "
        f"extracted_samples={result.stats.extracted_samples} "
        f"exporter_outputs={result.stats.exporter_outputs}"
    )
    return 0


def _handle_validate(config_path: Path) -> int:
    resolved = load_app_config(config_path)
    summary = validate_app_config(resolved)
    print(f"config_ok: {resolved.config_path}")
    print(
        "summary: "
        f"valid_samples={summary.valid_samples} "
        f"class_names={len(summary.class_names)} "
        f"extractor={summary.extractor_name} "
        f"exporter={summary.exporter_name}"
    )
    return 0


def _handle_list() -> int:
    capabilities = list_builtin_capabilities()
    print("Built-in extractors:")
    for item in capabilities["extractors"]:
        print(f"  - {item}")
    print("Built-in exporters:")
    for item in capabilities["exporters"]:
        print(f"  - {item}")
    print("Presets:")
    for name, preset in capabilities["presets"].items():
        print(
            f"  - {name}: extractor={preset['extractor']} exporter={preset['exporter']}"
        )
    return 0


class ValidationSummary:
    def __init__(
        self,
        *,
        valid_samples: int,
        class_names: list[str],
        extractor_name: str,
        exporter_name: str,
    ) -> None:
        self.valid_samples = valid_samples
        self.class_names = class_names
        self.extractor_name = extractor_name
        self.exporter_name = exporter_name


def validate_app_config(resolved: ResolvedAppConfig) -> ValidationSummary:
    if not resolved.dataset_dir.exists() or not resolved.dataset_dir.is_dir():
        raise ValidationFailure(
            f"dataset.path does not exist: '{resolved.dataset_dir}'"
        )

    report = inspect_dataset(resolved.dataset_dir, DEFAULT_IMAGE_FILE)
    if report.total_sample_dirs == 0:
        raise ValidationFailure(
            f"dataset directory '{resolved.dataset_dir}' has no sample folders"
        )
    if report.invalid_sample_count:
        details = "; ".join(
            f"{issue.sample_id}: {', '.join(issue.problems)}"
            for issue in report.issues[:5]
        )
        raise ValidationFailure(
            "dataset contains invalid samples; fix the dataset before export. "
            f"Examples: {details}"
        )

    pipeline_config = build_pipeline_export_config(resolved)
    samples = discover_samples(pipeline_config.dataset_dir, pipeline_config.image_file)
    extractor, _ = load_extractor(
        pipeline_config.extractor_spec, pipeline_config.extractor_opts
    )
    exporter, _ = load_exporter(
        pipeline_config.exporter_spec, pipeline_config.exporter_opts
    )
    missing = set(exporter.required_features) - set(extractor.produced_features)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValidationFailure(
            f"selected task/output combination is invalid; missing features: {joined}"
        )

    extraction_ctx = ExtractionContext(
        dataset_dir=pipeline_config.dataset_dir,
        image_file=pipeline_config.image_file,
        samples=samples,
        framework_options={},
    )
    dataset_info = extractor.describe_dataset(extraction_ctx)
    if not dataset_info.class_names:
        raise ValidationFailure("config must produce at least one class")

    return ValidationSummary(
        valid_samples=len(samples),
        class_names=list(dataset_info.class_names),
        extractor_name=resolved.extractor_spec,
        exporter_name=resolved.exporter_spec,
    )


def _resolve_output_config_path(output_config: Path) -> Path:
    resolved = output_config.resolve()
    if resolved.exists() and resolved.is_dir():
        return resolved / DEFAULT_CONFIG_PATH
    if resolved.suffix:
        return resolved
    return resolved / DEFAULT_CONFIG_PATH


def _print_inspect_report(report: InspectReport) -> None:
    print(f"Dataset: {report.dataset_dir}")
    print(f"Image file: {report.image_file}")
    print(
        "Samples: "
        f"total={report.total_sample_dirs} "
        f"valid={report.valid_sample_count} "
        f"invalid={report.invalid_sample_count}"
    )
    print("Scene tags:")
    if report.scene_tags:
        for item in report.scene_tags:
            print(f"  - {item.tag}: {item.count}")
    else:
        print("  - none")
    print("Object tags:")
    if report.object_tags:
        for item in report.object_tags:
            print(f"  - {item.tag}: {item.count}")
    else:
        print("  - none")
    print("Top object tag combinations:")
    if report.object_tag_combinations:
        for combination in report.object_tag_combinations:
            print(f"  - {', '.join(combination.tags)}: {combination.count}")
    else:
        print("  - none")
    print("Suggested class mappings:")
    if report.suggested_class_mappings:
        for suggestion in report.suggested_class_mappings:
            print(f"  - {suggestion.class_name}: {suggestion.count}")
    else:
        print("  - none")
    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"  - {issue.sample_id}: {', '.join(issue.problems)}")
    if report.recommendations:
        print("Recommendations:")
        for recommendation in report.recommendations:
            print(f"  - {recommendation}")


def _prompt_choice(prompt: str, options: list[tuple[str, str]], default: str) -> str:
    option_text = ", ".join(f"{key}={value}" for key, value in options)
    while True:
        answer = (
            input(f"{prompt} [{option_text}] (default {default}): ").strip() or default
        )
        for key, value in options:
            if answer == key or answer == value:
                return value
        print("Please choose one of the listed options.")


def _prompt_output_format(task: str) -> str:
    formats = ["preview"]
    if task in {"detection", "both"}:
        formats.append("yolo_bbox")
    if task in {"segmentation", "both"}:
        formats.append("yolo_seg")

    options = [(str(index), value) for index, value in enumerate(formats, start=1)]
    default = str(len(options))
    return _prompt_choice("Select output format", options, default=default)


def _select_builtin_extractor_spec(task: str) -> str:
    extractor_by_task = {
        "detection": "default-bbox",
        "segmentation": "default-seg",
        "both": "default-seg-bbox",
    }
    return cast(str, extractor_by_task[task])


def _select_builtin_exporter_spec(task: str, output_format: str) -> str:
    if output_format == "preview":
        return "default-preview-bbox" if task == "detection" else "default-preview-seg"
    if output_format == "yolo_bbox":
        return "default-yolo-bbox"
    return "default-yolo-seg"


def _build_builtin_exporter_options(
    output_format: str, include_empty: bool
) -> dict[str, object]:
    options: dict[str, object] = {"include_empty": include_empty}
    if output_format == "preview":
        options["compare_original"] = True
    else:
        options["splits"] = {"train": 0.8, "val": 0.2}
    return options


def _prompt_tag_selection(report: InspectReport) -> list[str]:
    if not report.object_tags:
        answer = _prompt_text(
            "No object tags were detected. Enter a class name", "example"
        )
        return [answer]

    default_indexes = ",".join(
        str(index) for index in range(1, len(report.object_tags) + 1)
    )
    while True:
        answer = input(
            "Select class tags by number or tag name, comma-separated "
            f"(default {default_indexes}): "
        ).strip()
        if not answer:
            return [item.tag for item in report.object_tags]

        selected: list[str] = []
        for raw_item in answer.split(","):
            item = raw_item.strip()
            if not item:
                continue
            if item.isdigit():
                index = int(item) - 1
                if 0 <= index < len(report.object_tags):
                    selected.append(report.object_tags[index].tag)
                    continue
            matching = next(
                (tag.tag for tag in report.object_tags if tag.tag == item), None
            )
            if matching is not None:
                selected.append(matching)
                continue
            print(f"Unknown tag selection: {item}")
            break
        else:
            unique_selected = list(dict.fromkeys(selected))
            if unique_selected:
                return unique_selected
        print("Please select one or more listed tags.")


def _prompt_text(prompt: str, default: str) -> str:
    answer = input(f"{prompt} (default {default}): ").strip()
    return answer or default


def _prompt_yes_no(prompt: str, default: str) -> bool:
    if default not in {"y", "n"}:
        _unreachable()
    while True:
        answer = (
            input(f"{prompt} [y/n] (default {default}): ").strip().lower() or default
        )
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _unreachable() -> NoReturn:
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
