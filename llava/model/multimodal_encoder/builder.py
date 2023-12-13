import os
from .clip_encoder import CLIPVisionTower, MetaTransformer, VideoProjector, UniTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower_name = getattr(vision_tower_cfg, 'vision_tower_name', None)
    is_absolute_path_exists = os.path.exists(vision_tower)

    # should be to args: vision_tower for name, vision_tower_path for model path
    if vision_tower_name == 'clip' and is_absolute_path_exists:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower_name == 'meta' and is_absolute_path_exists:
        return MetaTransformer(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower_name == 'video' and is_absolute_path_exists:
        return VideoProjector(vision_tower, args=vision_tower_cfg, **kwargs)
    if vision_tower_name == 'multiple' and is_absolute_path_exists:
        return UniTower(vision_tower, args=vision_tower_cfg, **kwargs)

    #if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        # return MetaTransformer(vision_tower, args=vision_tower_cfg, **kwargs)
        #return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
