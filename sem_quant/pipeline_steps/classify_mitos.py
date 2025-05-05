# sem_quant/pipeline_steps/classify_mitos.py

"""
Classifies processed mitochondria using a pre-trained FastAI model.

Loads the unclassified mitochondria DataFrame, extracts image crops,
runs inference using the specified classifier, and saves the DataFrame
with added prediction columns.
"""

import argparse
import os
import sys
import time
import gc
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import torch
from loguru import logger
from skimage.color import gray2rgb
from tifffile import imread # Keep for potential direct loading if needed
from tqdm import tqdm

# --- Import FastAI (optional dependency) ---
try:
    from fastai.learner import load_learner
    FASTAI_AVAILABLE = True
except ImportError:
    logger.error("fastai library not found. Please install it to run classification.")
    FASTAI_AVAILABLE = False
    # Define dummy load_learner if fastai is not available to avoid NameError later
    def load_learner(path, cpu=True):
        raise ImportError("fastai library not found. Cannot load learner.")

# --- Import shared components ---
try:
    from sem_quant.utils import setup_logging, smart_path
    from sem_quant.load_config import load_config, PipelineConfig
    from sem_quant.data_utils import load_image_dask
    from sem_quant.fastai_utils import label_func, get_device
    UTILS_AVAILABLE = True
except ImportError as e:
    # If basic utils fail, it's a critical error
    print(f"Error: Could not import necessary functions from 'sem_quant'. "
          f"Please ensure the package is installed or accessible. Error: {e}")
    sys.exit(1) # Exit if essential utils are missing

# --- Constants ---
LOGS_SUBDIR = 'logs'
ANNOTATION_SUFFIX = ".pkl"
# Suffix for the input file from process_mitos.py
UNCLASSIFIED_SUFFIX = "_unclassified"

def run_mitos_classification(config: PipelineConfig):
    """
    Main classification pipeline for mitochondria.

    Args:
        config: Loaded pipeline configuration object.
    """
    if not FASTAI_AVAILABLE:
         logger.error("FastAI library is required but not installed. Cannot proceed.")
         return # Exit if fastai is missing

    start_time = time.time()
    logger.info("--- Starting Mitochondria Classification ---")

    # --- Setup Paths and Parameters ---
    try:
        paths = config.paths
        data_props = config.data_properties
        mitos_classifier = config.mitos_classifier

        analysis_dir = Path(smart_path(paths.analysis_dir))
        im_path_str = smart_path(paths.im_path)
        output_prefix = paths.output_prefix
        mitos_data_suffix = paths.mitos_data_suffix # Base suffix for mito files

        # Define input (unclassified) and output (classified) filenames
        unclassified_filename = f"{output_prefix}{mitos_data_suffix}{UNCLASSIFIED_SUFFIX}{ANNOTATION_SUFFIX}"
        unclassified_filepath = analysis_dir / unclassified_filename
        # Final output will overwrite/use the standard mito suffix
        classified_filename = f"{output_prefix}{mitos_data_suffix}{ANNOTATION_SUFFIX}"
        classified_filepath = analysis_dir / classified_filename

        classifier_path = Path(smart_path(mitos_classifier.path))
        crop_pad = mitos_classifier.pad # Padding around centroid for cropping

    except Exception as e:
        logger.error(f"Error accessing configuration parameters: {e}", exc_info=True)
        return

    # --- Load Unclassified Mitochondria Data ---
    logger.info(f"Loading unclassified mitochondria data from: {unclassified_filepath}")
    if not unclassified_filepath.exists():
        logger.error(f"Input mitochondria data file not found: {unclassified_filepath}")
        logger.error("Please run the 'process_mitos.py' script first.")
        return
    try:
        df_mitos = pd.read_pickle(unclassified_filepath)
        if df_mitos.empty:
             logger.warning(f"Input mitochondria DataFrame is empty: {unclassified_filepath}. No classification to perform.")
             return
        logger.info(f"Loaded {len(df_mitos)} mitochondria for classification.")
    except Exception as e:
        logger.error(f"Failed to load mitochondria data from {unclassified_filepath}: {e}", exc_info=True)
        return

    # --- Load Image Data (for Cropping) ---
    logger.info(f"Loading base image data for cropping from: {im_path_str} at resolution {data_props.mitos_res}")
    try:
        # Load the dask array for the required resolution
        im_da, im_shape = load_image_dask(im_path_str, data_props.mitos_res)
        logger.info(f"Image loaded with shape: {im_shape}")
    except (FileNotFoundError, KeyError, Exception) as e:
        logger.error(f"Could not load image data from {im_path_str} at resolution {data_props.mitos_res}: {e}")
        return # Cannot proceed without image data

    # --- Load Classifier Model ---
    logger.info(f"Loading FastAI classifier model from: {classifier_path}")
    if not classifier_path.exists():
        logger.error(f"Classifier model file not found: {classifier_path}")
        return
    try:
        # Suppress UserWarning about pickle insecurity if desired
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            learn = load_learner(classifier_path, cpu=False) # Load directly to potential GPU
        logger.info("Classifier loaded successfully.")
        # Determine device and move model
        device = get_device()
        learn.model.to(device)
        logger.info(f"Classifier moved to device: {device}")

    except Exception as e:
        logger.error(f"Failed to load classifier model from {classifier_path}: {e}", exc_info=True)
        return

    # --- Classify Each Mitochondrion ---
    logger.info(f"Starting classification of {len(df_mitos)} mitochondria...")
    # Initialize prediction columns
    df_mitos['prediction'] = None
    df_mitos['prediction_prob'] = None
    df_mitos['prediction_prob'] = df_mitos['prediction_prob'].astype(object) # Allow storing lists/arrays

    classification_errors = 0
    # Check required columns for cropping
    if not all(c in df_mitos.columns for c in ['centroid-0', 'centroid-1']):
        logger.error("Missing 'centroid-0' or 'centroid-1' columns in input DataFrame. Cannot perform classification.")
        return

    for ind, row in tqdm(df_mitos.iterrows(), total=len(df_mitos), desc="Classifying Mitos"):
        try:
            # Calculate crop boundaries
            row_center = int(row['centroid-0'])
            col_center = int(row['centroid-1'])
            row_start = max(0, row_center - crop_pad)
            row_stop = min(im_shape[0], row_center + crop_pad)
            col_start = max(0, col_center - crop_pad)
            col_stop = min(im_shape[1], col_center + crop_pad)

            # Extract and compute the crop from the Dask array
            # Ensure crop size is correct (2*pad x 2*pad) handling edges
            im_crop_raw = im_da[row_start:row_stop, col_start:col_stop].compute()

            # Handle potential edge cases where crop is smaller than expected
            # Pad if necessary to ensure consistent input size for the model (e.g., 100x100 if pad=50)
            expected_size = 2 * crop_pad
            pad_h = expected_size - im_crop_raw.shape[0]
            pad_w = expected_size - im_crop_raw.shape[1]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                # Pad with edge values or a constant (e.g., 0) - edge might be better
                im_crop_padded = np.pad(im_crop_raw, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
                logger.trace(f"Padded crop for mito {row.get('label', ind)} from {im_crop_raw.shape} to {im_crop_padded.shape}")
            else:
                im_crop_padded = im_crop_raw

            # Convert to RGB 
            im_mito_rgb = gray2rgb(im_crop_padded)

            # Predict using the loaded learner
            cat, _, probs = learn.predict(im_mito_rgb)

            # Store results - use .loc/.at for safer assignment
            df_mitos.loc[ind, 'prediction'] = str(cat)
            # Store probabilities as a list (more universally serializable than tensor)
            df_mitos.at[ind, 'prediction_prob'] = probs.cpu().numpy().tolist() # Move to CPU, convert to numpy, then list

        except Exception as e:
            logger.warning(f"Failed to classify mitochondrion index {ind} (Label: {row.get('label', 'N/A')}): {e}", exc_info=False) # Log less verbose info
            logger.debug(f"Detailed error for index {ind}", exc_info=True) # Log full traceback at debug level
            classification_errors += 1
            df_mitos.loc[ind, 'prediction'] = "CLASSIFICATION_ERROR"
            df_mitos.at[ind, 'prediction_prob'] = None
            continue # Skip to the next mitochondrion

    if classification_errors > 0:
         logger.warning(f"Encountered {classification_errors} errors during classification.")

    logger.info("Finished classification loop.")

    # --- Save Final DataFrame ---
    try:
        logger.info(f"Saving final classified mitochondria DataFrame to: {classified_filepath}")
        # Ensure prediction_prob column is compatible with pickle
        # It should be list of floats now, which is fine.
        df_mitos.to_pickle(classified_filepath)
    except Exception as e:
        logger.error(f"Failed to save final DataFrame to {classified_filepath}: {e}", exc_info=True)

    # Optional: Clean up GPU memory
    if device == torch.device("cuda"):
        del learn
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleaned up GPU memory.")

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    logger.info("--- Mitochondria Classification Script Finished ---")


# --- Main Execution ---

if __name__ == "__main__":

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Classify mitochondria using a FastAI model."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the pipeline configuration JSON file."
    )
    args = parser.parse_args()

    # --- Load Config ---
    try:
        config_file_path = Path(smart_path(args.config_path))
        print(f"Loading configuration from: {config_file_path}") # Print before logging setup
        if not config_file_path.is_file():
             print(f"FATAL: Configuration file not found at '{config_file_path}'")
             sys.exit(1)
        config = load_config(str(config_file_path))
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load configuration '{args.config_path}': {e}")
        sys.exit(1)

    # --- Setup Logging using Config ---
    try:
        analysis_dir = Path(smart_path(config.paths.analysis_dir))
        log_dir = analysis_dir / LOGS_SUBDIR
        log_dir.mkdir(parents=True, exist_ok=True)
        # Use a specific log file name for this step
        log_file_path = log_dir / f"classify_mitos_{{time}}.log"
        print(f"Setting up logging. Log file pattern: {log_file_path}")
        setup_logging(str(log_file_path), console_level="INFO", file_level="DEBUG")
    except Exception as e:
         print(f"WARNING: Error setting up logging based on config paths: {e}. Logging might be incomplete.")
         logger.remove() # Remove default handler if exists
         logger.add(sys.stderr, level="INFO") # Basic console logger
         logger.warning("File logging setup failed. Using console logging only.")

    # --- Run Main Logic ---
    try:
        run_mitos_classification(config) 
    except Exception as e:
        # Catch any unhandled exceptions from the main processing function
        logger.exception("Pipeline step 'run_mitos_classification' failed with an unhandled exception.")
        sys.exit(1) # Exit with error status

    # --- Successful Exit ---
    sys.exit(0)