from pathlib import Path

from rv_ds.ir import DatasetIR, InstanceRecord, SampleRecord
from rv_ds.plugin_api import PluginOptions
from rv_ds.sdk import (
    extract_bbox,
    extract_largest_polygon,
    load_scene_meta,
    normalize_bbox,
    object_mask,
    parse_classes_file,
    read_index_map,
    resolve_class_from_tags,
)


class ExtractorOptions(PluginOptions):
    classes_file: Path
    target_tags: list[str] = []
    mode: str = "segment"


def build_extractor(opts: ExtractorOptions):
    classes = parse_classes_file(opts.classes_file)
    target_tags = set(opts.target_tags)
    mode = opts.mode

    class ExtractorImpl:
        def extract_dataset(self, ctx):
            samples = []
            for sample in ctx.samples:
                scene = load_scene_meta(sample.meta_path)
                index_map = read_index_map(sample.index_path)
                sample_instances = []

                for obj in scene.objects:
                    if target_tags and not (set(obj.tags) & target_tags):
                        continue

                    class_name, class_id = resolve_class_from_tags(
                        list(obj.tags), classes
                    )
                    if class_id is None:
                        continue

                    mask = object_mask(index_map, obj.index)
                    bbox = extract_bbox(mask)
                    poly = (
                        extract_largest_polygon(mask)
                        if mode in ("segment", "both")
                        else None
                    )
                    norm_bbox = (
                        normalize_bbox(bbox, index_map.shape[1], index_map.shape[0])
                        if bbox is not None
                        else None
                    )

                    sample_instances.append(
                        InstanceRecord(
                            sample_id=sample.sample_id,
                            object_index=obj.index,
                            class_name=class_name,
                            class_id=class_id,
                            object_tags=list(obj.tags),
                            bbox_xyxy=bbox,
                            bbox_norm_cxcywh=norm_bbox,
                            polygon_norm=poly,
                            area_px=int(mask.sum()),
                            extra={},
                        )
                    )

                samples.append(
                    SampleRecord(
                        sample_id=sample.sample_id,
                        scene_tags=list(scene.tags),
                        image_src_path=sample.image_path,
                        image_out_name=f"{sample.sample_id}.png",
                        width=index_map.shape[1],
                        height=index_map.shape[0],
                        instances=sample_instances,
                        extra={},
                    )
                )

            return DatasetIR(samples=samples, class_names=classes, meta={"custom": True})

    return ExtractorImpl()
