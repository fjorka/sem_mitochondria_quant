from loguru import logger
import torch
from omegaconf import OmegaConf
from pathlib import Path

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import SAM2 components: {e}. SAM functionality will be unavailable.")
    SAM2_AVAILABLE = False
    build_sam2 = None
    SAM2AutomaticMaskGenerator = None

def get_pytorch_device():
    """Determines the appropriate PyTorch device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA device found. Using GPU.")
        # Optional: Configure TF32 for Ampere GPUs (as in notebooks)
        if torch.cuda.get_device_properties(0).major >= 8:
            logger.info("Ampere GPU detected. Allowing TF32.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        logger.info("CUDA device not found. Using CPU.")
    return device

def setup_sam_mask_generator(model_config_path: str, checkpoint_path: str, seg_params, device: torch.device):
    """
    Initializes SAM2 model and mask generator with specific parameters.

    Args:
        model_config_path: Path to the SAM model config YAML.
        checkpoint_path: Path to the SAM model checkpoint (.pt).
        seg_params: Pydantic model object containing segmentation parameters
                    (e.g., config.axons_segmentation or config.mitos_segmentation).
        device: The torch device (CPU or CUDA).

    Returns:
        Initialized SAM2AutomaticMaskGenerator instance or None if SAM2 is unavailable.
    """
    if not SAM2_AVAILABLE:
         logger.error("SAM2 components are not available. Cannot setup mask generator.")
         return None

    logger.info(f"Setting up SAM model from config: {model_config_path} and checkpoint: {checkpoint_path}")
    logger.info(f"Using segmentation parameters: {seg_params.dict()}") # Log the parameters being used

    try:
        # apply_postprocessing=False as seen in notebooks
        sam_model = build_sam2(model_config_path, checkpoint_path, device=device, apply_postprocessing=False)
        logger.info("SAM2 model built successfully.")

        # Dynamically get parameters from the passed seg_params object
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam_model,
            points_per_side=getattr(seg_params, 'points_per_side', 32), # Provide defaults if needed
            points_per_batch=getattr(seg_params, 'points_per_batch', 64),
            pred_iou_thresh=getattr(seg_params, 'pred_iou_thresh', 0.88),
            stability_score_thresh=getattr(seg_params, 'stability_score_thresh', 0.95),
            stability_score_offset=getattr(seg_params, 'stability_score_offset', 1.0),
            crop_n_layers=getattr(seg_params, 'crop_n_layers', 0),
            box_nms_thresh=getattr(seg_params, 'box_nms_thresh', 0.7),
            crop_n_points_downscale_factor=getattr(seg_params, 'crop_n_points_downscale_factor', 1),
            min_mask_region_area=getattr(seg_params, 'min_mask_region_area', 0),
            use_m2m=getattr(seg_params, 'use_m2m', False)
        )
        logger.info("SAM2AutomaticMaskGenerator initialized successfully.")
        return mask_generator

    except Exception as e:
        logger.error(f"Failed to setup SAM model or mask generator: {e}", exc_info=True)
        return None