from pydantic import BaseModel
from typing import Optional


# ---------- Shared Base Components ----------

class Paths(BaseModel):
    im_path: str
    analysis_dir: str
    output_prefix: str
    exclude_file_suffix: str
    axons_data_suffix: str
    mitos_data_suffix: str


class DataProperties(BaseModel):
    px_size: int
    mitos_res: int
    axons_res: int
    row_offset: int
    col_offset: int


class SamModel(BaseModel):
    model_cfg: str
    sam2_checkpoint: str


# ---------- Axon Components ----------

class AxonsSegmentationParams(BaseModel):
    tile_size_x: int
    tile_size_y: int
    tile_overlap_x: int
    tile_overlap_y: int
    points_per_side: int
    points_per_batch: int
    pred_iou_thresh: float
    stability_score_thresh: float
    stability_score_offset: float
    crop_n_layers: int
    box_nms_thresh: float
    crop_n_points_downscale_factor: int
    min_mask_region_area: int
    use_m2m: bool


class AxonsFilters(BaseModel):
    min_area: int
    max_area: int
    max_solidity: float
    iou_threshold: float
    small_object_area_threshold: int 
    tile_edge_padding: int        
    soma_edge_padding: int 


# ---------- Mitochondria Components ----------

class MitosSegmentationParams(BaseModel):
    points_per_side: int
    points_per_batch: int
    pred_iou_thresh: float
    stability_score_thresh: float
    stability_score_offset: float
    crop_n_layers: int
    box_nms_thresh: float
    crop_n_points_downscale_factor: int
    min_mask_region_area: int
    use_m2m: bool


class MitosClassifier(BaseModel):
    path: str
    pad: int


class MitosFilters(BaseModel):
    min_area: int
    max_area: int
    max_eccentricity: float
    min_iou: float
    iou_threshold: float
    pad: int


# ---------- Top-Level Config ----------

class PipelineConfig(BaseModel):
    paths: Paths
    data_properties: DataProperties
    sam_model: SamModel
    axons_segmentation: AxonsSegmentationParams
    axons_filters: AxonsFilters
    mitos_segmentation: MitosSegmentationParams
    mitos_classifier: MitosClassifier
    mitos_filters: MitosFilters


def load_config(path: str) -> PipelineConfig:
    import json
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return PipelineConfig(**config_dict)
