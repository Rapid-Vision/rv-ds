from dataclasses import dataclass
from pathlib import Path

from .errors import SampleFailure, ValidationFailure


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    root: Path
    meta_path: Path
    index_path: Path
    image_path: Path


def discover_samples(dataset_dir: Path, image_file: str) -> list[SamplePaths]:
    if not dataset_dir.exists():
        raise ValidationFailure(f"dataset directory does not exist: '{dataset_dir}'")
    if not dataset_dir.is_dir():
        raise ValidationFailure(f"dataset path is not a directory: '{dataset_dir}'")

    sample_dirs = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name
    )
    if not sample_dirs:
        raise ValidationFailure(
            f"dataset directory '{dataset_dir}' has no sample folders"
        )

    samples: list[SamplePaths] = []
    for sample_dir in sample_dirs:
        meta_path = sample_dir / "_meta.json"
        index_path = sample_dir / "IndexOB.png"
        image_path = sample_dir / image_file

        missing = [
            path.name
            for path in (meta_path, index_path, image_path)
            if not path.exists() or not path.is_file()
        ]
        if missing:
            raise SampleFailure(
                sample_dir, f"missing required files: {', '.join(missing)}"
            )

        samples.append(
            SamplePaths(
                sample_id=sample_dir.name,
                root=sample_dir,
                meta_path=meta_path,
                index_path=index_path,
                image_path=image_path,
            )
        )

    return samples
