# sem_quant/log_utils.py
import sys
import os
from loguru import logger
import re
from pathlib import Path
import platform
import ntpath

# Define standard formats as constants
CONSOLE_LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
FILE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"

def setup_logging(log_file_path, console_level="INFO", file_level="DEBUG"):
    """Configures Loguru with console and a specific file sink."""
    logger.remove()
    logger.add(
        sys.stderr, level=console_level, format=CONSOLE_LOG_FORMAT,
        colorize=True, enqueue=True
    )
    try:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir: # Ensure directory path is not empty
             os.makedirs(log_dir, exist_ok=True)

        logger.add(
            log_file_path, rotation="10 MB", level=file_level,
            format=FILE_LOG_FORMAT, enqueue=True
        )
        logger.info(f"Logging setup complete. Console: {console_level}, File: {file_level} ('{log_file_path}')")
        return True
    except Exception as e:
        logger.error(f"Failed to add file logger to path '{log_file_path}': {e}")
        # Still log to console, but indicate file logging failed
        logger.info(f"Logging setup partially complete (Console only). Console: {console_level}")
        return False

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

def running_in_wsl():
    """Detect if the current environment is WSL."""
    return 'microsoft-standard' in platform.uname().release or 'WSL' in platform.version()

def windows_to_wsl_path_manual(path):
    """
    Convert a Windows-style path (e.g., 'D:\\folder\\file') to a WSL path (e.g., '/mnt/d/folder/file').

    Parameters:
        path (str): A Windows path string.

    Returns:
        str: The corresponding WSL path string.
    """
    # recognizes Windows paths
    if len(path) >= 2 and path[1] == ':' and path[0].isalpha():
        drive, tail = ntpath.splitdrive(path)
        drive_letter = drive[0].lower()
        unix_path = tail.replace('\\', '/')
        return f"/mnt/{drive_letter}{unix_path}"
    else:
        return path

def smart_path(path):
    """Return a WSL-formatted path if running inside WSL, otherwise return original."""
    if running_in_wsl():
        return windows_to_wsl_path_manual(path)
    return path