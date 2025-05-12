"""
Orchestrates the execution of the SEM Mitochondria Quantification pipeline steps.

Reads a configuration file for pipeline parameters, sets up logging,
and runs each pipeline script in the defined sequence using Python environments
specified *within this script*, checking for errors at each step.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from loguru import logger
import os 
import importlib.resources

# Attempt to import shared utilities from the 'sem_quant' package
try:
    from sem_quant.load_config import load_config, PipelineConfig # Or just load_config if it returns a dict
    from sem_quant.utils import setup_logging, smart_path
except ImportError as e:
    print(f"Error: Failed to import modules from 'sem_quant'. "
          f"Ensure 'sem_quant' package is installed or accessible "
          f"in the Python path. Error: {e}")
    sys.exit(1)

# --- Constants ---
LOGS_SUBDIR = 'logs'

PIPELINE_STEPS = [
    "segment_axons.py",
    "process_axons.py",
    "segment_mitos.py",
    "process_mitos.py",
    "classify_mitos.py"
]
# Base directory for the step scripts within the package
STEPS_SUBPACKAGE_DIR = "pipeline_steps"

# --- Environment Configuration (Define Paths Here) ---

# Set the default Python executable path.
# Set to None to use the same environment as this script.
DEFAULT_PYTHON_ENV: str | None = "/h20/home/lab/miniconda3/envs/sam-env/bin/python"

# Define specific Python environments for individual steps (optional overrides).
# Keys are the script filenames from PIPELINE_STEPS.
# Values are paths to the Python executables for those steps.
STEP_PYTHON_ENVS: dict[str, str] = {
    "classify_mitos.py": r"/h20/home/lab/miniconda3/envs/fastai-kasia/bin/python"
}


def get_python_executable(
    step_script_name: str,
    default_env_path: str | None,
    step_envs: dict[str, str] | None
) -> str:
    """
    Determines the appropriate Python executable path for a given step.

    Args:
        step_script_name: Filename of the step script.
        default_env_path: Path to the default Python executable, if specified.
        step_envs: Dictionary mapping step script names to their specific Python executables.

    Returns:
        The path to the Python executable to use.
    """
    # Prioritize step-specific environment
    if step_envs and step_script_name in step_envs:

        specific_path = step_envs[step_script_name]
        logger.debug(f"Using step-specific environment for {step_script_name}: {specific_path}")
        return specific_path

    # Fallback to default environment
    elif default_env_path:

        logger.debug(f"Using default environment for {step_script_name}: {default_env_path}")
        return default_env_path

    # Fallback to the current environment
    logger.debug(f"Using current environment for {step_script_name}: {sys.executable}")
    return sys.executable


def run_step(
    step_script_name: str,
    config_path_str: str,
    sem_quant_root: Path,
    default_env_python: str | None,
    step_envs: dict[str, str] | None
) -> bool:
    """
    Runs a single pipeline step script as a subprocess using the designated Python environment.

    Args:
        step_script_name: The filename of the step script (e.g., "segment_axons.py").
        config_path_str: The absolute path to the configuration JSON file (for pipeline params).
        sem_quant_root: The absolute path to the 'sem_quant' package directory.
        default_env_python: Path to the default Python executable, if configured.
        step_envs: Dictionary mapping step scripts to specific Python executables.

    Returns:
        True if the step completed successfully (exit code 0), False otherwise.
    """
    logger.info(f"--- Starting Step: {step_script_name} ---")
    step_start_time = time.time()

    script_path = sem_quant_root / STEPS_SUBPACKAGE_DIR / step_script_name

    try:
        if not script_path.is_file():
            logger.error(f"Pipeline step script not found: {script_path}")
            return False

        # Determine the Python executable using the passed-in constants
        python_executable = get_python_executable(step_script_name, default_env_python, step_envs)

        command = [
            python_executable,
            str(script_path),
            config_path_str
        ]
        logger.debug(f"Executing command: {' '.join(command)}")

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        log_level_stdout = "INFO" if process.returncode == 0 else "DEBUG"
        if process.stdout:
            logger.log(log_level_stdout, f"Stdout from {step_script_name}:\n{process.stdout.strip()}")
        if process.stderr:
            logger.warning(f"Stderr from {step_script_name}:\n{process.stderr.strip()}")

        if process.returncode != 0:
            logger.error(f"Step {step_script_name} failed with exit code {process.returncode}.")
            return False
        else:
            duration = time.time() - step_start_time
            logger.info(f"--- Step {step_script_name} completed successfully in {duration:.2f}s ---")
            return True

    except FileNotFoundError:
        logger.error(f"Could not find Python interpreter ('{python_executable}') or script: {script_path}")
        return False
    except Exception as e:
        logger.exception(f"An unexpected error occurred while trying to run {step_script_name}: {e}")
        return False


def run_pipeline():
    
    """Parses arguments, loads config, sets up logging, and runs the pipeline steps."""
    parser = argparse.ArgumentParser(
        description="Run the SEM Mitochondria Quantification pipeline using environments defined in this script."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the pipeline configuration JSON file (for analysis parameters)."
    )
    args = parser.parse_args()

    # --- Validate Environment Paths (Optional but Recommended) ---
    validated_default_env = DEFAULT_PYTHON_ENV
    if DEFAULT_PYTHON_ENV and not Path(DEFAULT_PYTHON_ENV).is_file():
        print(f"WARNING: Default Python environment executable not found: {DEFAULT_PYTHON_ENV}. "
              "Will attempt to fall back to current environment or step-specific ones.")

    # changes provided dict of step envs to only include those that exist
    validated_step_envs = {}
    for step, path in STEP_PYTHON_ENVS.items():
        if not Path(path).is_file():
             print(f"WARNING: Step-specific Python executable for '{step}' not found: {path}. "
                   "This step will use the default or current environment.")
        else:
             validated_step_envs[step] = path

    # --- Load Configuration (for pipeline parameters) ---
    config = None 
    try:
        config_file_input_path = Path(smart_path(args.config_path))
        if not config_file_input_path.is_file():
            print(f"FATAL: Configuration file not found: {config_file_input_path}")
            sys.exit(1)

        # Load the pipeline parameters config
        config = load_config(str(config_file_input_path)) # Returns PipelineConfig or dict
        config_file_abs_path = str(config_file_input_path.resolve())
        print(f"Pipeline parameters loaded successfully from: {config_file_abs_path}")

    except Exception as e:
        print(f"FATAL: Failed to load configuration '{args.config_path}': {e}")
        sys.exit(1)

    # --- Setup Logging ---
    try:
        if hasattr(config, 'paths') and hasattr(config.paths, 'analysis_dir'):
            analysis_dir_str = config.paths.analysis_dir
        else:
            # Handle case where config structure is unexpected
            print("FATAL: Cannot determine 'analysis_dir' from loaded configuration.")
            sys.exit(1)

        analysis_dir = Path(smart_path(analysis_dir_str))
        log_dir = analysis_dir / LOGS_SUBDIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / f"pipeline_run_{{time}}.log"
        print(f"Setting up pipeline orchestrator logging. Log file: {log_file_path}")
        setup_logging(str(log_file_path), console_level="INFO", file_level="DEBUG")
    except AttributeError:
         print(f"FATAL: Could not access expected attributes ('paths', 'analysis_dir') on the loaded config object.")
         sys.exit(1)
    except KeyError as e:
         print(f"FATAL: Missing expected key in configuration dictionary for logging setup: {e}.")
         sys.exit(1)
    except Exception as e:
        print(f"WARNING: Error setting up file logging to '{log_file_path}': {e}. Logging to console only.")
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # --- Determine Package Path ---
    sem_quant_package_root: Path | None = None # Initialize
    try:
        # Dynamically find the path to the installed 'sem_quant' package directory
        # importlib.resources.files returns a Traversable object representing the package directory
        package_files_ref = importlib.resources.files('sem_quant')

        # Convert the Traversable to an absolute Path object
        sem_quant_package_root = Path(package_files_ref).resolve()

        # Optional: Basic sanity check (might be redundant if importlib found it)
        if not sem_quant_package_root.is_dir() or not (sem_quant_package_root / "__init__.py").is_file():
             logger.error(f"FATAL: Dynamically located path '{sem_quant_package_root}' does not appear "
                          f"to be the 'sem_quant' package directory (__init__.py missing or not a dir).")
             sys.exit(1)

        logger.info(f"Dynamically located 'sem_quant' package root: {sem_quant_package_root}")

    except ModuleNotFoundError:
        logger.error("FATAL: The 'sem_quant' package was not found in the current Python environment.")
        logger.error(f"Please ensure 'sem_quant' is installed in the environment running this script: "
                     f"'{sys.executable}'. You might need to run 'pip install .' or 'pip install -e .' "
                     f"in the project directory.")
        sys.exit(1)

    except Exception as e:
         # Log the full exception details for debugging
         logger.exception(f"FATAL: An unexpected error occurred while dynamically locating the 'sem_quant' package: {e}")
         sys.exit(1)

    # --- Execute Pipeline Steps ---
    logger.info("=== Starting SEM Mitochondria Quantification Pipeline Execution ===")
    pipeline_start_time = time.time()

    for step_script in PIPELINE_STEPS:
        success = run_step(
            step_script,
            config_file_abs_path,
            sem_quant_package_root, # package root
            validated_default_env, # script's default env constant
            validated_step_envs  # script's step env constant dict
        )
        if not success:
            logger.error(f"Pipeline execution failed during step: {step_script}")
            logger.info("=== Pipeline Execution FAILED ===")
            sys.exit(1) # Exit with non-zero code on failure

    pipeline_duration = time.time() - pipeline_start_time
    logger.info(f"=== Pipeline Execution Completed Successfully in {pipeline_duration:.2f}s ===")
    sys.exit(0) # Exit with zero code on success

if __name__ == "__main__":
    run_pipeline()