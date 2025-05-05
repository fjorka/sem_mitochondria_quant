import platform
import ntpath
import re
import dask.array as da
from skimage.io import imread
from loguru import logger
from pathlib import Path
from .utils import smart_path


def load_image_dask(im_path_str: str, resolution_level: int):
    """
    Loads a specific resolution level from an image file (OME-TIFF/Zarr store) as a Dask array.

    Args:
        im_path_str: Path to the image file.
        resolution_level: The resolution level index (component) to load.

    Returns:
        A tuple containing:
            - dask.array.Array: The image data as a Dask array.
            - tuple: The shape of the loaded image data.
        Returns (None, None) if loading fails.

    Raises:
        FileNotFoundError: If the image file does not exist.
        KeyError: If the specified resolution level is not found.
        Exception: For other potential loading errors.
    """
    im_path = Path(smart_path(im_path_str)) # Use smart_path if needed
    logger.info(f"Attempting to load image data from: {im_path} at resolution level {resolution_level}")
    try:
        if not im_path.exists():
            raise FileNotFoundError(f"Image file not found at: {im_path}")

        store = imread(im_path, aszarr=True)
        # Access the specific resolution level using 'component'
        im_da = da.from_zarr(store, resolution_level)
        im_shape = im_da.shape
        logger.info(f"Image level {resolution_level} loaded as Dask array with shape: {im_shape}")
        return im_da, im_shape

    except FileNotFoundError as e:
        logger.error(e)
        raise
    except KeyError:
        logger.error(f"Resolution level {resolution_level} not found in Zarr store at {im_path}. Check file structure.")
        raise
    except Exception as e:
        logger.error(f"Failed to load image data from {im_path}: {e}", exc_info=True)
        raise

def tile_image(image, tile_size_x=1024, tile_size_y=1024, tile_overlap_x=300, tile_overlap_y=300):
    """
    Tiles an image into overlapping patches.
    
    Args:
        image (np.ndarray): The input image (H, W, C) or (H, W).
        tile_size_x (int): Width of each tile.
        tile_size_y (int): Height of each tile.
        tile_overlap_x (int): Overlap between tiles horizontally.
        tile_overlap_y (int): Overlap between tiles vertically.
        
    Returns:
        tiles (list of np.ndarray): List of image tiles.
        coords (list of tuple): List of (x_start, y_start, x_end, y_end) for each tile.
    """
    tiles = []
    coords = []

    H, W = image.shape[:2]

    stride_x = tile_size_x - tile_overlap_x
    stride_y = tile_size_y - tile_overlap_y

    for y in range(0, H, stride_y):
        for x in range(0, W, stride_x):
            x_start = x
            y_start = y
            x_end = min(x + tile_size_x, W)
            y_end = min(y + tile_size_y, H)

            # Adjust start if we're at the edge and the tile is smaller
            if x_end - x_start < tile_size_x:
                x_start = max(0, W - tile_size_x)
                x_end = W
            if y_end - y_start < tile_size_y:
                y_start = max(0, H - tile_size_y)
                y_end = H

            tile = image[y_start:y_end, x_start:x_end]
            tiles.append(tile)
            coords.append((x_start, y_start, x_end, y_end))

    return tiles, coords

