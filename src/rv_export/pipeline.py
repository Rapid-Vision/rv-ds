import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from .classes import ClassResolver, parse_classes_file
from .filters import (
    object_passes_target_tags,
    parse_csv_tags,
    passes_min_counts,
    scene_passes_filters,
)
from .geometry import bbox_from_mask, largest_polygon_from_mask, normalize_bbox
from .mask_ops import object_mask, read_index_map
from .models import SceneObject, load_scene_meta
from .scanner import discover_samples
from .yolo import format_detect_line, format_segment_line, write_data_yaml

TaskName = Literal["detect", "segment", "both"]


@dataclass(frozen=True)
class ExportConfig:
    dataset_dir: Path
    output_dir: Path
    classes_path: Path
    image_file: str
    task: TaskName
    target_tags: set[str]
    min_count: dict[str, int]
    require_tags: set[str]
    exclude_tags: set[str]
    include_empty: bool
    format_name: str = "yolo"


@dataclass
class ExportStats:
    processed: int = 0
    exported: int = 0
    skipped: int = 0
    errors: int = 0
    annotations_detect: int = 0
    annotations_segment: int = 0


@dataclass
class ExportResult:
    export_dir: Path
    stats: ExportStats


def build_config(
    dataset_dir: Path,
    output_dir: Path,
    classes_path: Path,
    image_file: str,
    task: TaskName,
    target_tags_csv: str | None,
    min_count: dict[str, int],
    require_tags_csv: str | None,
    exclude_tags_csv: str | None,
    include_empty: bool,
    format_name: str,
) -> ExportConfig:
    return ExportConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        classes_path=classes_path,
        image_file=image_file,
        task=task,
        target_tags=parse_csv_tags(target_tags_csv),
        min_count=min_count,
        require_tags=parse_csv_tags(require_tags_csv),
        exclude_tags=parse_csv_tags(exclude_tags_csv),
        include_empty=include_empty,
        format_name=format_name,
    )


def run_export(config: ExportConfig) -> ExportResult:
    class_names = parse_classes_file(config.classes_path)
    resolver = ClassResolver(class_names)
    samples = discover_samples(config.dataset_dir, config.image_file)

    export_dir = _build_export_dir(config.output_dir)
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)

    stats = ExportStats()

    for sample in samples:
        stats.processed += 1
        scene = load_scene_meta(sample.meta_path)
        if not scene_passes_filters(scene, config.require_tags, config.exclude_tags):
            stats.skipped += 1
            continue

        selected: list[tuple[SceneObject, int]] = []
        for obj in scene.objects:
            if not object_passes_target_tags(obj, config.target_tags):
                continue
            class_id = resolver.resolve_object_class(obj)
            if class_id is None:
                continue
            selected.append((obj, class_id))

        if not passes_min_counts([obj for obj, _ in selected], config.min_count):
            stats.skipped += 1
            continue

        index_map = read_index_map(sample.index_path)
        label_lines: list[str] = []

        for obj, class_id in selected:
            mask = object_mask(index_map, obj.index)

            if config.task in ("detect", "both"):
                bbox = bbox_from_mask(mask)
                if bbox is not None:
                    line = format_detect_line(
                        class_id=class_id,
                        bbox=normalize_bbox(
                            bbox, width=index_map.shape[1], height=index_map.shape[0]
                        ),
                    )
                    label_lines.append(line)
                    stats.annotations_detect += 1

            if config.task in ("segment", "both"):
                polygon = largest_polygon_from_mask(mask)
                if polygon is not None:
                    line = format_segment_line(class_id=class_id, polygon=polygon)
                    label_lines.append(line)
                    stats.annotations_segment += 1

        if not label_lines and not config.include_empty:
            stats.skipped += 1
            continue

        shutil.copy2(sample.image_path, images_dir / f"{sample.sample_id}.png")
        (labels_dir / f"{sample.sample_id}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        stats.exported += 1

    write_data_yaml(export_dir / "data.yaml", class_names)
    _write_meta(export_dir / "rv_export_meta.json", config, stats)

    return ExportResult(export_dir=export_dir, stats=stats)


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


def _write_meta(path: Path, config: ExportConfig, stats: ExportStats) -> None:
    payload = {
        "config": {
            **asdict(config),
            "dataset_dir": str(config.dataset_dir),
            "output_dir": str(config.output_dir),
            "classes_path": str(config.classes_path),
            "target_tags": sorted(config.target_tags),
            "require_tags": sorted(config.require_tags),
            "exclude_tags": sorted(config.exclude_tags),
        },
        "stats": asdict(stats),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
