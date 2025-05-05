"""

"""

import argparse
import os
import time
import gc
import pickle as pkl
from pathlib import Path
import warnings
import dask.array as da
import numpy as np
import pandas as pd
import torch # Keep torch import if device object is used directly
from loguru import logger
from skimage.color import gray2rgb
from skimage.draw import polygon
from tifffile import imread
from tqdm import tqdm

# --- Import shared components ---
try:
    from sem_quant.utils import setup_logging, smart_path
    from sem_quant.load_config import load_config
    from sem_quant.data_utils import load_image_dask, tile_image
    from sem_quant.sam_utils import get_pytorch_device, setup_sam_mask_generator
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Error: Could not import necessary functions from 'sem_quant'. Please ensure it's installed and contains 'config_loader.py', 'data_utils.py', and 'sam_utils.py'. Error: {e}")
    UTILS_AVAILABLE = False
    exit(1)

# --- Constants ---
ANNOTATION_SUFFIX = ".pkl"
LOGS_SUBDIR = 'logs'

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='sam2.sam2_image_predictor')

# --- Function Definitions ---


def main(config):
    """Main execution function for axon segmentation."""

    # --- Basic Setup ---
    try:
        # Determine device using the utility function
        device = get_pytorch_device()

        # Get paths and parameters from config
        paths = config.paths
        sam_model_config = config.sam_model
        mitos_seg_params = config.mitos_segmentation # Specific params for this step

        analysis_dir = Path(smart_path(paths.analysis_dir))
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        mitos_subdir = Path(smart_path(paths.analysis_dir)) / paths.mitos_data_suffix
        mitos_subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analysis directory set to: {analysis_dir}")

        # define resolution adjustment factor
        data_props = config.data_properties
        res_adjust_power = data_props.axons_res - data_props.mitos_res
        res_adjust_factor = 2**res_adjust_power
    
    except Exception as e:
         logger.error(f"Error during setup: {e}", exc_info=True)
         return

    # --- Load and Prepare Data ---
    try:
        im_da, _ = load_image_dask(paths.im_path, data_props.mitos_res)

    except Exception as e:
         logger.error(f"Error loading image data: {e}", exc_info=True)
         return

    # --- Initialize Model using sam_utils ---
    try:
        mask_generator = setup_sam_mask_generator(
            sam_model_config.model_cfg,
            sam_model_config.sam2_checkpoint,
            mitos_seg_params, # Pass the axon-specific parameters
            device
        )
        if mask_generator is None:
             logger.error("Mask generator initialization failed. Exiting.")
             return
        logger.info("SAM mask generator initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SAM model: {e}", exc_info=True)
        return
    
    # --- Load Axon Data ---
    try:
        axon_data_filename = config.paths.output_prefix + config.paths.axons_data_suffix + ANNOTATION_SUFFIX
        axon_data_path = analysis_dir / axon_data_filename
        logger.info(f"Loading axon data from: {axon_data_path}")
        
        if not axon_data_path.exists():
            logger.error(f"Axon data file not found at {axon_data_path}. "
                         "Ensure the axon segmentation step ran successfully.")
            return
        
        df_axons = pd.read_pickle(axon_data_path)
        logger.info(f"Loaded axon DataFrame with {len(df_axons)} entries.")
        
        # Check for required columns (adjust based on processing_utils output)
        required_cols = ['label', 'inside_bbox-0', 'inside_bbox-1', 'inside_bbox-2', 'inside_bbox-3']
        if not all(col in df_axons.columns for col in required_cols):
             logger.error(f"Axon DataFrame is missing required columns. Needed: {required_cols}")
             # Log existing columns for debugging
             logger.debug(f"Existing columns: {df_axons.columns.tolist()}")
             return

    except Exception as e:
        logger.error(f"Error loading axon data from {axon_data_path}: {e}", exc_info=True)
        return

    # --- Segment mitos ---
    logger.info("Starting mitochondria segmentation loop...")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Use tqdm for progress bar
    for index, row in tqdm(df_axons.iterrows(), total=len(df_axons), desc="Segmenting Mitos"):
        try:
            # Check if 'inside_bbox-0' etc. are present and not NaN before proceeding
            if pd.isna(row['inside_bbox-0']):
                logger.warning(f"Skipping axon label {row['label']}: Missing 'inside_bbox' data.")
                skipped_count += 1
                continue

            # Calculate bounding box coordinates adjusted for resolution and offset
            row_start = int((row['inside_bbox-0'] + data_props.row_offset) * res_adjust_factor)
            row_end = int((row['inside_bbox-2'] + data_props.row_offset) * res_adjust_factor)
            col_start = int((row['inside_bbox-1'] + data_props.col_offset) * res_adjust_factor)
            col_end = int((row['inside_bbox-3'] + data_props.col_offset) * res_adjust_factor)

            if row_start >= row_end or col_start >= col_end:
                 logger.warning(f"Skipping axon label {row['label']}: Invalid bounding box after adjustments "
                                f"([{row_start}:{row_end}, {col_start}:{col_end}]). Original was "
                                f"([{row['inside_bbox-0']}:{row['inside_bbox-2']}, {row['inside_bbox-1']}:{row['inside_bbox-3']}])")
                 skipped_count += 1
                 continue

            # Extract the image region as a NumPy array
            # Using .compute() to bring the Dask chunk into memory for SAM
            im_region = im_da[row_start:row_end, col_start:col_end].compute()

            if im_region.size == 0:
                logger.warning(f"Skipping axon label {row['label']}: Extracted image region is empty.")
                skipped_count +=1
                continue

            # Convert grayscale to RGB for SAM
            im_rgb = gray2rgb(im_region)

            # Generate masks using SAM
            masks_generated = mask_generator.generate(im_rgb)

            # Define output path for the pickle file
            output_filename = f"{paths.output_prefix}{str(row['label']).zfill(6)}_{paths.mitos_data_suffix}{ANNOTATION_SUFFIX}"
            output_path = mitos_subdir / output_filename

            # Save the generated masks
            with open(output_path, 'wb') as f:
                pkl.dump(masks_generated, f)

            processed_count += 1

            # Optional: Clear memory periodically
            if index % 1 == 0: # Adjust frequency as needed
                gc.collect()
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing axon label {row.get('label', 'UnknownLabel')} at index {index}: {e}", exc_info=True)
            error_count += 1
            continue

    logger.info("--- Mitos Segmentation Script Finished ---")

if __name__ == "__main__":
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Segment mitos...")
    parser.add_argument("config_path", type=str, help="Path to config JSON.")
    args = parser.parse_args()

    # --- Load Config ---
    try:
        print(f"Loading configuration from: {args.config_path}")
        config = load_config(args.config_path)
    except Exception as e:
        print(f"FATAL: Failed to load configuration '{args.config_path}': {e}")
        exit(1)

    # --- Setup Logging using Config ---
    try:
        analysis_dir = Path(smart_path(config.paths.analysis_dir))
        log_dir = analysis_dir / LOGS_SUBDIR
        log_file_path = log_dir / f"segment_mitos_{{time}}.log"
    except Exception as e:
         print(f"WARNING: Error determining log path from config: {e}.")
         exit(1)

    setup_logging(log_file_path=log_file_path)

    # --- Run Main Logic ---
    try:
        main(config) # Pass the loaded config
    except Exception as e:
        logger.exception("Pipeline step failed with an unhandled exception.")