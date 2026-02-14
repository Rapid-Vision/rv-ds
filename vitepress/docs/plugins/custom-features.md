# Custom Features

Custom feature contracts allow extractor/exporter extensions beyond built-in fields.

## Naming

Rules:

- lowercase alphanumeric plus `_:.`
- must start with `custom:` unless feature is a standard built-in constant

Example: `custom:instance_bbox_6d`

## Declaring Contracts

Extractor:

```python
produced_features = frozenset({"instance_class", "custom:instance_bbox_6d"})
```

Exporter:

```python
required_features = frozenset({"custom:instance_bbox_6d"})
```

## Payload Storage

Use extension dictionaries:

- dataset-level: `dataset.meta["custom:<feature>"]`
- sample-level: `sample.extra["custom:<feature>"]`
- instance-level: `instance.extra["custom:<feature>"]`
