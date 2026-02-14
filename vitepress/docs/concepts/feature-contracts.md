# Feature Contracts

Before extraction begins, framework checks compatibility:

`exporter.required_features ⊆ extractor.produced_features`

If any required feature is missing, run fails fast with a contract mismatch error.

## Standard Features

- `sample_class`
- `instance_class`
- `instance_segment`
- `instance_bbox`

## Custom Features

Custom names must:

- use lowercase `[a-z0-9_:.]`
- start with `custom:` if non-standard

Example: `custom:instance_bbox_6d`

## Payload Placement

- dataset-level: `dataset.meta["custom:<feature>"]`
- sample-level: `sample.extra["custom:<feature>"]`
- instance-level: `instance.extra["custom:<feature>"]`
