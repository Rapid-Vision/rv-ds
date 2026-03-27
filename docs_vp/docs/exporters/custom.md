# Writing A Custom Exporter

A custom exporter is a Python file referenced by `exporter.spec`.

```yaml
exporter:
  spec: ./plugins/custom-exporter.py
  options: {}
```

`rv-ds` loads the file and looks for a class named `ExporterPlugin`.

## Requirements

- Inherit `BaseExporter`
- Set `OptionsModel`
- Declare non-empty `required_features`
- Implement `export_dataset(ctx)`

## Minimal Example

```python
import json
from pathlib import Path

from rv_ds.plugin_api import (
    INSTANCE_BBOX,
    INSTANCE_CLASS,
    BaseExporter,
    ExporterRunResult,
    PluginOptions,
)


class ExporterOptions(PluginOptions):
    pass


class ExporterPlugin(BaseExporter[ExporterOptions]):
    OptionsModel = ExporterOptions
    required_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})

    def export_dataset(self, ctx):
        labels = []
        for sample in ctx.iter_samples(order="sequential"):
            for inst in sample.instances:
                if inst.class_id is None or inst.bbox_norm_cxcywh is None:
                    continue
                labels.append({"sample_id": sample.sample_id, "bbox": list(inst.bbox_norm_cxcywh)})

        labels_path = ctx.write_text(Path("labels.json"), json.dumps(labels, indent=2))
        return ExporterRunResult(outputs=[str(labels_path)], stats={"labels": len(labels)})
```

## Export Context

Use `ctx` to work safely inside the output directory:

- `ctx.iter_samples(...)`: stream extracted samples
- `ctx.mkdir(path)`: create a directory inside the output root
- `ctx.write_text(path, text)`: write a text file inside the output root
- `ctx.copy_image(src, dst)`: copy images into the output root
- `ctx.warn(message)`: add a non-fatal warning

Do not write outside `ctx.output_dir`. The framework rejects that.

## Required Features

Use the feature constants from `rv_ds.plugin_api`:

- `INSTANCE_CLASS`
- `INSTANCE_BBOX`
- `INSTANCE_SEGMENT`

Declare only the features your exporter actually reads.

## Example File

Full example plugin: `examples/plugins/custom-exporter.py`
