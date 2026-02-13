import argparse
import sys
from pathlib import Path
from typing import cast

from .errors import RVExportError, ValidationFailure
from .pipeline import TaskName, build_config, run_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rv-export", description="Export RV datasets to YOLO format"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export a dataset")
    export_parser.add_argument("dataset_dir", type=Path)
    export_parser.add_argument("--format", required=True, choices=["yolo"])
    export_parser.add_argument(
        "--task", default="both", choices=["detect", "segment", "both"]
    )
    export_parser.add_argument("--classes", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, default=Path("./exports"))
    export_parser.add_argument("--image-file", default="Image.png")
    export_parser.add_argument("--target-tags")
    export_parser.add_argument("--min-count", action="append", default=[])
    export_parser.add_argument("--require-tags")
    export_parser.add_argument("--exclude-tags")
    export_parser.add_argument("--include-empty", action="store_true")

    return parser


def parse_min_count(items: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValidationFailure(
                f"invalid --min-count value '{item}', expected format '<tag>=<number>'"
            )

        tag, raw_count = item.split("=", 1)
        tag = tag.strip()
        raw_count = raw_count.strip()

        if not tag:
            raise ValidationFailure(
                f"invalid --min-count value '{item}': tag cannot be empty"
            )

        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValidationFailure(
                f"invalid --min-count value '{item}': count must be integer"
            ) from exc

        if count < 0:
            raise ValidationFailure(
                f"invalid --min-count value '{item}': count cannot be negative"
            )

        result[tag] = count

    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "export":
        parser.error("unsupported command")

    try:
        config = build_config(
            dataset_dir=args.dataset_dir,
            output_dir=args.output,
            classes_path=args.classes,
            image_file=args.image_file,
            task=cast(TaskName, args.task),
            target_tags_csv=args.target_tags,
            min_count=parse_min_count(args.min_count),
            require_tags_csv=args.require_tags,
            exclude_tags_csv=args.exclude_tags,
            include_empty=args.include_empty,
            format_name=args.format,
        )
        result = run_export(config)
    except RVExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"export_dir={result.export_dir}")
    print(
        "summary: "
        f"processed={result.stats.processed} "
        f"exported={result.stats.exported} "
        f"skipped={result.stats.skipped} "
        f"errors={result.stats.errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
