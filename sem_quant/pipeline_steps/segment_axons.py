"""
This script segments axons in an image using the SAM2 model.
It loads the image, applies annotations, initializes the SAM model
using parameters specific to axon segmentation from the config file,
tiles the image, processes each tile to generate masks, filters them based
on area (within SAM generator), and saves the results for each tile.

Requires an environment with SAM2 installed.
Assumes existence of 'sem_quant' package with config_loader, utils, and sam_utils.
"""

import argparse
import os
import sys
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
OUTPUT_FILENAME_TEMPLATE = "tile_masks_{row_start}_{col_start}.pkl"

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module='sam2.sam2_image_predictor')

# --- Function Definitions ---

def apply_annotations(
    image: np.ndarray, analysis_dir_str: str, exclude_name: str
) -> np.ndarray:
    """Loads annotations and applies them by zeroing out annotated areas."""
    annotation_file_path = Path(analysis_dir_str) / f"{exclude_name}{ANNOTATION_SUFFIX}"
    logger.info(f"Attempting to load annotations from: {annotation_file_path}")

    if not annotation_file_path.exists():
        logger.warning(f"Annotation file not found at {annotation_file_path}. Skipping annotation application.")
        return image

    try:
        with open(annotation_file_path, "rb") as f:
            annotations = pkl.load(f)
        logger.info(f"Loaded {len(annotations)} annotation polygon(s).")

        # Create a copy to avoid modifying the original image array
        image_annotated = image.copy()
        # Draw the polygons by setting pixels to 0
        for i, poly in enumerate(annotations):
            try:
                # Get row and column coordinates for the polygon
                rr, cc = polygon(poly[:, 0], poly[:, 1], image_annotated.shape)
                # Set the pixels within the polygon to 0
                image_annotated[rr, cc] = 0
            except IndexError as e:
                 # Log a warning if polygon coordinates are out of bounds
                 logger.warning(f"Skipping annotation polygon {i+1} due to potential coordinate issue (IndexError): {e}")
            except Exception as e:
                 logger.warning(f"Error processing annotation polygon {i+1}: {e}", exc_info=True)

        logger.info("Annotations applied successfully.")
        return image_annotated
    except Exception as e:
        logger.error(f"Failed to load or apply annotations from {annotation_file_path}: {e}", exc_info=True)
        # Return the original image if annotation loading/application fails
        return image

def process_tiles(
    image: np.ndarray,
    mask_generator, # an initialized SAM2AutomaticMaskGenerator instance
    config, # Pass config for tile parameters and output path formatting
):
    """Tiles the image, processes each tile for masks using the provided generator, and saves them."""

    # get analysis directory from config
    analysis_dir = Path(smart_path(config.paths.analysis_dir))
    output_dir = analysis_dir / config.paths.axons_data_suffix

    # Extract tiling parameters from the axon-specific config section
    tile_params = config.axons_segmentation
    logger.info(f"Tiling with parameters: Size=({tile_params.tile_size_x}, {tile_params.tile_size_y}), Overlap=({tile_params.tile_overlap_x}, {tile_params.tile_overlap_y})")

    try:
        tiles, coords = tile_image(
            image,
            tile_params.tile_size_x, tile_params.tile_size_y,
            tile_params.tile_overlap_x, tile_params.tile_overlap_y
        )
        logger.info(f"Generated {len(tiles)} tiles.")
    except Exception as e:
        logger.error(f"Failed during image tiling: {e}", exc_info=True)
        raise # Reraise exception as tiling is critical

    num_tiles = len(tiles)
    processed_tiles = 0
    skipped_tiles = 0

    for tile, coord in tqdm(zip(tiles, coords), total=num_tiles, desc="Processing Axon Tiles"):
        
        # Create a unique ID for logging/filenames using padded coordinates
        tile_id = f"{str(coord[1]).zfill(5)}_{str(coord[0]).zfill(5)}" # row_col format

        try:
            # Skip processing if the tile is entirely black (e.g., outside image or fully annotated)
            if np.max(tile) == 0:
                 logger.trace(f"Tile {tile_id} is empty (max value is 0). Skipping.")
                 skipped_tiles += 1
                 continue

            tile_rgb = gray2rgb(tile)
            masks = mask_generator.generate(tile_rgb)
            logger.trace(f"Generated {len(masks)} raw masks for tile {tile_id}.")

            # Convert masks to DataFrame - Filtering based on area is now done inside SAM generator via min_mask_region_area
            # Further filtering (like max_area) happens in the next script (process_axons)
            df_tile = pd.DataFrame(masks)

            if not df_tile.empty:
                # Add tile coordinate information to each mask's record
                df_tile["tile_row_start"] = coord[1]
                df_tile["tile_row_end"] = coord[3]
                df_tile["tile_col_start"] = coord[0]
                df_tile["tile_col_end"] = coord[2]

                # Define output filename using the template and padded coordinates
                file_name = OUTPUT_FILENAME_TEMPLATE.format(row_start=str(coord[1]).zfill(5), col_start=str(coord[0]).zfill(5))
                output_path = output_dir / file_name
                logger.trace(f"Saving {len(df_tile)} masks for tile {tile_id} to {output_path}")
                try:
                     df_tile.to_pickle(output_path)
                except Exception as e:
                     logger.error(f"Failed to save masks for tile {tile_id} to {output_path}: {e}", exc_info=True)
            else:
                 logger.trace(f"No masks met the generator's criteria for tile {tile_id}.")

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(2) 
            processed_tiles += 1

        except Exception as e:
            # Log errors encountered during processing a specific tile
            logger.error(f"Error processing tile {tile_id}: {e}", exc_info=True)
            # Decide whether to continue or stop on error. Here, we log and continue.

    logger.info(f"Tile processing finished. Processed: {processed_tiles}, Skipped (empty): {skipped_tiles}")

def run_axon_segmentation(config):
    """Main execution function for axon segmentation."""

    # --- Basic Setup ---
    try:
        # Determine device using the utility function
        device = get_pytorch_device()

        # Get paths and parameters from config
        paths = config.paths
        data_props = config.data_properties
        sam_model_config = config.sam_model
        axon_seg_params = config.axons_segmentation # Specific params for this step

        analysis_dir = Path(smart_path(paths.analysis_dir))
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        axons_subdir = Path(smart_path(paths.analysis_dir)) / paths.axons_data_suffix
        axons_subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Analysis directory set to: {analysis_dir}")
    
    except Exception as e:
         logger.error(f"Error during setup: {e}", exc_info=True)
         return

    # --- Load and Prepare Data ---
    try:
        im_da, _ = load_image_dask(paths.im_path, data_props.axons_res)
        im_org = im_da.compute()
        logger.info("Image computed.")
        # Construct exclude name from prefix and suffix in config
        exclude_name = f'{paths.output_prefix}{paths.exclude_file_suffix}'
        im_annotated = apply_annotations(im_org, str(analysis_dir), exclude_name)

    except Exception as e:
         logger.error(f"Error loading or preparing image data: {e}", exc_info=True)
         return

    # --- Initialize Model using sam_utils ---
    try:
        mask_generator = setup_sam_mask_generator(
            sam_model_config.model_cfg,
            sam_model_config.sam2_checkpoint,
            axon_seg_params, # Pass the axon-specific parameters
            device
        )
        if mask_generator is None:
             logger.error("Mask generator initialization failed. Exiting.")
             return
        logger.info("SAM mask generator initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SAM model: {e}", exc_info=True)
        return

    # --- Process Tiles ---
    try:
        process_tiles(im_annotated, mask_generator, config)
    except Exception as e:
        logger.error(f"An error occurred during tile processing: {e}", exc_info=True)
        return

    logger.info("--- Axon Segmentation Script Finished ---")

if __name__ == "__main__":
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Segment axons...")
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
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"classify_mitos_{{time}}.log"
    except Exception as e:
         print(f"WARNING: Error determining log path from config: {e}.")
         exit(1)

    setup_logging(log_file_path=log_file_path)

    # --- Run Main Logic ---
    try:
        run_axon_segmentation(config) # Pass the loaded config
    except Exception as e:
        logger.exception("Pipeline step failed with an unhandled exception.")

    # --- Successful Exit ---
    sys.exit(0)