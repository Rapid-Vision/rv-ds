# Options and Validation

## `PluginOptions` baseline

`PluginOptions` uses strict Pydantic config:

- `strict=True`
- `extra="forbid"`

This means wrong types and unknown keys fail fast.

## Loader validation sequence

1. Validate `produced_features` or `required_features` declaration.
2. Validate `OptionsModel` inheritance.
3. Validate options payload via `model_validate`.
4. Instantiate plugin class.
5. Verify required methods exist.

## Common Validation Failures

- missing plugin file
- plugin file not `.py`
- missing `ExtractorPlugin`/`ExporterPlugin`
- wrong base class
- invalid options schema/value
- non-empty feature declaration missing
- invalid feature name format

See `/reference/errors` for trigger/fix catalog.
