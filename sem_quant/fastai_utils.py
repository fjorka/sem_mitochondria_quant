import re
from pathlib import Path
import torch
from loguru import logger


def label_func(o):
    """
    Extracts a label from a filename based on a specific naming convention.

    The function expects the input `o` to represent a file path with a name 
    matching the pattern: "<label>_XXXXXX.tif", where <label> is an arbitrary 
    string without underscores, and XXXXXX is a 6-digit number.

    Parameters:
        o (str or Path): A path to the file whose name will be parsed.

    Returns:
        str: The extracted label from the filename.

    Raises:
        AssertionError: If the filename does not match the expected pattern.
    """
    pat = re.compile(r'^([^_]+)_\d{6}\.tif$')

    o = Path(o)
    match = pat.match(o.name)
    assert match, f'Pattern failed on {o.name}'
    label = match.group(1)
    return label  # "discard" if label == "discard" else "mito"

def get_device() -> torch.device:
    """Determines the appropriate PyTorch device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA device found. Using GPU for classification.")
    else:
        device = torch.device("cpu")
        logger.info("CUDA device not found. Using CPU for classification.")
    return device