# Data Model

## Terminology

- **Dataset**: full export payload (`DatasetIR` for dumps)
- **Sample**: one scene/image item (`SampleRecord`)
- **Instance**: one object annotation (`InstanceRecord`)

## Core Fields

Sample contains:

- `sample_id`, `scene_tags`
- source/output image paths and dimensions
- `instances`
- extensible `extra`

Instance contains:

- class identity (`class_name`, `class_id`)
- geometry (`bbox_xyxy`, `bbox_norm_cxcywh`, `polygon_norm`)
- `object_index`, `object_tags`, `area_px`
- extensible `extra`

Field-level references:

- `/reference/schema-dataset`
- `/reference/schema-sample`
- `/reference/schema-instance`
