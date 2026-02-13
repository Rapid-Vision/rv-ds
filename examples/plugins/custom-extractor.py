from rv_ds.errors import ValidationFailure
from rv_ds.ir import DatasetIR, InstanceRecord, SampleRecord
from rv_ds.plugin_api import (
    INSTANCE_BBOX,
    INSTANCE_CLASS,
    BaseExtractor,
    PluginOptions,
)
from rv_ds.sdk import (
    extract_bbox,
    load_scene_meta,
    normalize_bbox,
    object_mask,
    read_index_map,
)


class ExtractorOptions(PluginOptions):
    tags: list[str]


class ExtractorPlugin(BaseExtractor[ExtractorOptions]):
    OptionsModel = ExtractorOptions
    produced_features = frozenset({INSTANCE_CLASS, INSTANCE_BBOX})

    def __init__(self, opts: ExtractorOptions) -> None:
        super().__init__(opts)
        self.opts = opts

    def extract_dataset(self, ctx):
        class_names = self.opts.tags
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        samples = []
        for sample in ctx.samples:
            scene = load_scene_meta(sample.meta_path)
            index_map = read_index_map(sample.index_path)
            instances = []

            for obj in scene.objects:
                if not obj.tags:
                    continue
                if len(obj.tags) > 1:
                    raise ValidationFailure(
                        f"sample '{sample.sample_id}' object index={obj.index} has multiple tags; "
                        "this simple extractor expects exactly one tag per object"
                    )

                tag = obj.tags[0]
                class_id = class_to_id.get(tag)
                if class_id is None:
                    continue

                mask = object_mask(index_map, obj.index)
                bbox_xyxy = extract_bbox(mask)
                if bbox_xyxy is None:
                    continue

                bbox_norm = normalize_bbox(
                    bbox_xyxy,
                    width=index_map.shape[1],
                    height=index_map.shape[0],
                )

                instances.append(
                    InstanceRecord(
                        sample_id=sample.sample_id,
                        object_index=obj.index,
                        class_name=tag,
                        class_id=class_id,
                        object_tags=[tag],
                        bbox_xyxy=bbox_xyxy,
                        bbox_norm_cxcywh=bbox_norm,
                        polygon_norm=None,
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
                    instances=instances,
                    extra={},
                )
            )

        return DatasetIR(
            samples=samples, class_names=class_names, meta={"custom": True}
        )
