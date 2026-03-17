# Writing A Custom Extractor

A custom extractor is a Python file referenced by `extractor.spec`.

```yaml
extractor:
  spec: ./plugins/custom-extractor.py
  options:
    tags: [sphere, cube]
```

`rv-ds` loads the file and looks for a class named `ExtractorPlugin`.

## Requirements

- Inherit `BaseExtractor`
- Set `OptionsModel`
- Declare non-empty `produced_features`
- Implement `describe_dataset(ctx)`
- Implement `extract_sample(ctx, sample)`

## Minimal Example

```python
from rv_ds.ir import InstanceRecord, SampleRecord
from rv_ds.plugin_api import (
    INSTANCE_BBOX,
    INSTANCE_CLASS,
    BaseExtractor,
    ExtractorDatasetInfo,
    PluginOptions,
)


class ExtractorOptions(PluginOptions):
    tags: list[str]


class ExtractorPlugin(BaseExtractor[ExtractorOptions]):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})

    def describe_dataset(self, ctx):
        return ExtractorDatasetInfo(class_names=self.opts.tags)

    def extract_sample(self, ctx, sample):
        return SampleRecord(
            sample_id=sample.sample_id,
            scene_tags=[],
            image_src_path=sample.image_path,
            image_out_name=f"{sample.sample_id}.png",
            width=0,
            height=0,
            instances=[],
            extra={},
        )
```

## What To Return

`describe_dataset(ctx)` must return:

- `class_names`: ordered class list used by exporters
- `meta`: optional extractor metadata

`extract_sample(ctx, sample)` must return:

- `SampleRecord` for samples you want to export
- `None` to skip a sample

## Produced Features

Use built-in feature constants from `rv_ds.plugin_api`:

- `INSTANCE_CLASS`
- `INSTANCE_BBOX`
- `INSTANCE_SEGMENT`

Declare exactly what your exporter needs. If an exporter requires a feature your extractor does not produce, `rv-ds validate` fails before export.

## Example File

Full example plugin: `examples/plugins/custom-extractor.py`
