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
import os # Import os for path validation if needed

# Attempt to import shared utilities from the 'sem_quant' package
try:
    # Assuming load_config still handles the *pipeline parameters* from the JSON
    from sem_quant.load_config import load_config, PipelineConfig # Or just load_config if it returns a dict
    from sem_quant.utils import setup_logging, smart_path
except ImportError as e:
    print(f"Error: Failed to import modules from 'sem_quant'. "
          f"Ensure 'sem_quant' package is installed or accessible "
          f"in the Python path. Error: {e}")
    sys.exit(1)

# --- Constants ---
LOGS_SUBDIR = 'logs'
# Define the sequence of pipeline step scripts relative to sem_quant/pipeline_steps/
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
# Example Linux/macOS: "/path/to/your/venv/bin/python"
# Example Windows: r"C:\path\to\your\venv\Scripts\python.exe"
DEFAULT_PYTHON_ENV: str | None = "/h20/home/lab/miniconda3/envs/sam-env/bin/python"
# Example: DEFAULT_PYTHON_ENV = "/home/user/venvs/sem_quant_default/bin/python"

# Define specific Python environments for individual steps (optional overrides).
# Keys are the script filenames from PIPELINE_STEPS.
# Values are paths to the Python executables for those steps.
STEP_PYTHON_ENVS: dict[str, str] = {
    "classify_mitos.py": r"/h20/home/lab/miniconda3/envs/fastai-kasia/bin/python"
    # Example: Override environment for segmentation steps
    # "segment_axons.py": "/path/to/segmentation_env/bin/python",
    # "segment_mitos.py": "/path/to/segmentation_env/bin/python",

    # Example: Use a different env for classification
    # "classify_mitos.py": r"C:\envs\tf_env\Scripts\python.exe",
}
# --- End Environment Configuration ---


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
        # Optional: Add validation here if desired
        # if not Path(specific_path).is_file():
        #     logger.warning(f"Step-specific Python env not found for {step_script_name} at {specific_path}. Falling back.")
        # else:
        logger.debug(f"Using step-specific environment for {step_script_name}: {specific_path}")
        return specific_path

    # Fallback to default environment
    if default_env_path:
        # Optional: Add validation here if desired
        # if not Path(default_env_path).is_file():
        #     logger.warning(f"Default Python env not found at {default_env_path}. Falling back.")
        # else:
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


def main():
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
        # Optionally set to None if you want fallback behavior on error:
        # validated_default_env = None

    validated_step_envs = {}
    for step, path in STEP_PYTHON_ENVS.items():
        if not Path(path).is_file():
             print(f"WARNING: Step-specific Python executable for '{step}' not found: {path}. "
                   "This step will use the default or current environment.")
        else:
             validated_step_envs[step] = path # Only add valid paths

    # --- Load Configuration (for pipeline parameters) ---
    config = None # Use PipelineConfig object or dict depending on load_config
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

    # --- Setup Logging for the Orchestrator ---
    try:
        # Access analysis_dir from the loaded config object/dict
        if isinstance(config, dict): # If load_config returns a dict
            analysis_dir_str = config.get("paths", {}).get("analysis_dir", ".")
        else: # Assuming it's an object like PipelineConfig
             # Check if 'paths' attribute exists and has 'analysis_dir'
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
    try:
        project_root = Path(__file__).resolve().parent.parent
        sem_quant_root = project_root / "sem_quant"
        if not (sem_quant_root / "__init__.py").is_file():
             logger.warning(f"Could not definitively locate 'sem_quant' package directory at: {sem_quant_root}. Assuming relative imports within steps will work.")
        logger.debug(f"Project root estimated as: {project_root}")
        logger.debug(f"Package 'sem_quant' path estimated as: {sem_quant_root}")
    except Exception as e:
         logger.error(f"Could not determine necessary paths: {e}")
         sys.exit(1)

    # --- Execute Pipeline Steps ---
    logger.info("=== Starting SEM Mitochondria Quantification Pipeline Execution ===")
    pipeline_start_time = time.time()

    for step_script in PIPELINE_STEPS:
        # Pass the environment constants defined in this script
        success = run_step(
            step_script,
            config_file_abs_path, # Pass path to param config
            sem_quant_root,
            validated_default_env,   # Pass the script's default env constant
            validated_step_envs      # Pass the script's step env constant dict
        )
        if not success:
            logger.error(f"Pipeline execution failed during step: {step_script}")
            logger.info("=== Pipeline Execution FAILED ===")
            sys.exit(1)

    pipeline_duration = time.time() - pipeline_start_time
    logger.info(f"=== Pipeline Execution Completed Successfully in {pipeline_duration:.2f}s ===")
    sys.exit(0)


if __name__ == "__main__":
    main()