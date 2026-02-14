# Plugin Development Overview

A plugin is a Python file exporting one class:

- extractor: `ExtractorPlugin`
- exporter: `ExporterPlugin`

Both must inherit the correct base class and declare non-empty feature contracts.

## Required class-level declarations

Extractor:

- `OptionsModel` subclassing `PluginOptions`
- `produced_features: frozenset[str]`

Exporter:

- `OptionsModel` subclassing `PluginOptions`
- `required_features: frozenset[str]`

## Runtime entrypoints

Extractor:

- `describe_dataset(ctx) -> ExtractorDatasetInfo`
- `extract_sample(ctx, sample) -> SampleRecord | None`

Exporter:

- `export_dataset(ctx) -> ExporterRunResult | dict[str, Any]`

Use pages in this section for exact method contracts and examples.
