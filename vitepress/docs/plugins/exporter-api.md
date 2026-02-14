# Exporter API

## Required Symbols

Plugin file must define:

- `ExporterOptions(PluginOptions)`
- `ExporterPlugin(BaseExporter)`

## Contract

```python
from rv_ds.plugin_api import BaseExporter, ExporterRunResult

class ExporterPlugin(BaseExporter):
    def export_dataset(self, ctx) -> ExporterRunResult | dict[str, Any]:
        ...
```

## `ExportContext`

Important fields and methods:

- `dataset_info`: class names + dataset meta
- `framework_options`
- `iter_samples(max_samples=None, order="sequential"|"random")`
- `safe_path(path)`
- `mkdir(path)`
- `write_text(path, content)`
- `copy_image(src, dst)`
- `add_output(path)`
- `warn(message)` / `warnings`

## Stream Rules

- one-pass only: calling `iter_samples` more than once fails
- `max_samples` must be positive when provided

## Return Value

Use `ExporterRunResult(stats, outputs, meta)` or an equivalent dict.

Dict form accepted:

- structured: `{"stats": ..., "outputs": ..., "meta": ...}`
- shorthand: integer values become `stats`, remaining keys become `meta`
