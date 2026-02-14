# Extractor API

## Required Symbols

Plugin file must define:

- `ExtractorOptions(PluginOptions)`
- `ExtractorPlugin(BaseExtractor)`

## Contract

```python
from rv_ds.plugin_api import BaseExtractor, ExtractorDatasetInfo
from rv_ds.ir import SampleRecord
from rv_ds.scanner import SamplePaths

class ExtractorPlugin(BaseExtractor):
    def describe_dataset(self, ctx) -> ExtractorDatasetInfo:
        ...

    def extract_sample(self, ctx, sample: SamplePaths) -> SampleRecord | None:
        ...
```

## `describe_dataset(ctx)`

Called once before exporting starts.

Must return `ExtractorDatasetInfo`:

- `class_names: list[str]` (strings, unique)
- `meta: dict[str, Any]`

## `extract_sample(ctx, sample)`

Called lazily by framework stream.

Return:

- `SampleRecord` to yield sample
- `None` to skip sample

Every returned sample is validated by framework (`SampleRecord.validate()`).

## Context Fields

`ExtractionContext` provides:

- `dataset_dir`
- `image_file`
- `samples` (all discovered `SamplePaths`)
- `framework_options` (`_framework` object)
- `warn(message)` / `warnings`
