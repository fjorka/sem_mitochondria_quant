import re
from pathlib import Path
import torch
from loguru import logger


pat = re.compile(r'^([^_]+)_\d{6}\.tif$')

def label_func(o):
    o = Path(o)
    match = pat.match(o.name)
    assert match, f'Pattern failed on {o.name}'
    label = match.group(1)
    return label

def get_device() -> torch.device:
    """Determines the appropriate PyTorch device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA device found. Using GPU for classification.")
    else:
        device = torch.device("cpu")
        logger.info("CUDA device not found. Using CPU for classification.")
    return device