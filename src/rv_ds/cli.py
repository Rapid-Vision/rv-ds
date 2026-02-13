import argparse
import sys
from pathlib import Path

from .errors import RVExportError
from .pipeline import ExportConfig, load_json_opts, run_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rv-ds", description="Two-stage RV export pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Extract and export a dataset")
    export_parser.add_argument("dataset_dir", type=Path)
    export_parser.add_argument("--extractor", required=True)
    export_parser.add_argument("--extractor-opts", type=Path)
    export_parser.add_argument("--exporter", required=True)
    export_parser.add_argument("--exporter-opts", type=Path)
    export_parser.add_argument("--output", "-o", type=Path, default=Path("./exports"))
    export_parser.add_argument("--image-file", default="Image.png")
    export_parser.add_argument("--fail-on-plugin-warning", action="store_true")
    export_parser.add_argument("--dump-ir", action="store_true")
    export_parser.add_argument("--debug", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "export":
        parser.error("unsupported command")

    try:
        config = ExportConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output,
            image_file=args.image_file,
            extractor_spec=args.extractor,
            extractor_opts=load_json_opts(args.extractor_opts),
            exporter_spec=args.exporter,
            exporter_opts=load_json_opts(args.exporter_opts),
            fail_on_plugin_warning=args.fail_on_plugin_warning,
            dump_ir=args.dump_ir,
        )
        result = run_export(config)
    except RVExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 2

    print(f"export_dir={result.export_dir}")
    print(
        "summary: "
        f"discovered_samples={result.stats.discovered_samples} "
        f"extracted_samples={result.stats.extracted_samples} "
        f"exporter_outputs={result.stats.exporter_outputs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
