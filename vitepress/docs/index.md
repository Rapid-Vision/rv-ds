---
layout: home

hero:
  name: "rv-ds"
  text: "Streaming export framework for RV datasets"
  tagline: "Build extractors and exporters with explicit feature contracts and strict validation."
  actions:
    - theme: brand
      text: Start Here
      link: /getting-started
    - theme: alt
      text: Plugin Development
      link: /plugins/overview

features:
  - title: Streaming pipeline
    details: Exporters pull sample records on demand via one-pass iteration.
  - title: Strict plugin options
    details: Pydantic strict mode with forbidden extra fields catches config mistakes early.
  - title: Contract-first compatibility
    details: Extractor produced features are validated against exporter required features before execution.
---

## Who This Is For

Primary audience: plugin developers building custom extractor and exporter plugins.

Secondary audience: CLI users running built-in pipelines for segmentation, detection, and previews.

## What You Get

- End-to-end CLI docs for `rv-ds export`
- Built-in extractor/exporter option references
- Plugin API contracts and lifecycle
- Schema and error references
- Troubleshooting and migration notes
