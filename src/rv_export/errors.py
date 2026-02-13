from pathlib import Path


class RVExportError(RuntimeError):
    """Base error type for user-facing export failures."""


class ValidationFailure(RVExportError):
    """Raised when user input or metadata is invalid."""


class SampleFailure(RVExportError):
    """Raised when a concrete sample directory cannot be processed."""

    def __init__(self, sample_path: Path, message: str) -> None:
        super().__init__(f"{sample_path}: {message}")
        self.sample_path = sample_path
        self.message = message
