import os
import re
import subprocess
import time

# Path to directory containing the profile_driver_*.py files
SCRIPT_DIR = "/home/work/persistent-kernels/tune_streamk/"
FAILED_CONFIGS_PATH = os.path.join(SCRIPT_DIR, "compile_driver.py.failed_configs")

# Regex to extract the failed kernel configuration from the error message
KERNEL_CONFIG_REGEX = r"(BM\d+_BN\d+_BK\d+_GM\d+_nW\d+_nS\d+_EU\d+_kP\d+_mfma\d+)"

# Sleep time between retries
SLEEP_TIME_BETWEEN_RUNS = 2  # seconds

def log_failed_config(config):
    """Logs the failed config to the failed_configs file."""
    with open(FAILED_CONFIGS_PATH, "a") as f:
        f.write(config + "\n")

def run_profile_script(script_path):
    """Run the profile script and capture any failure due to resource limits."""
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr  # Return the error output for analysis

def process_scripts():
    """Process each profile_driver_*.py script, rerun until there are no more failures."""
    # Get a list of all profile_driver_*.py files
    profile_files = [f for f in os.listdir(SCRIPT_DIR) if f.startswith("profile_driver_") and f.endswith(".py")]

    # Load already failed configurations from the file
    try:
        with open(FAILED_CONFIGS_PATH, "r") as f:
            failed_configs = set(cfg.strip() for cfg in f.readlines())
    except FileNotFoundError:
        failed_configs = set()

    for profile_file in profile_files:
        script_path = os.path.join(SCRIPT_DIR, profile_file)
        print(f"Running {script_path}...")

        while True:
            # Run the script and capture the output
            output = run_profile_script(script_path)

            # Check if the script produced an OutOfResources error and extract the failed kernel config
            if "triton.runtime.errors.OutOfResources" in output:
                match = re.search(KERNEL_CONFIG_REGEX, output)
                if match:
                    failed_kernel_config = match.group(1)
                    if failed_kernel_config not in failed_configs:
                        print(f"Logging failed config: {failed_kernel_config}")
                        log_failed_config(failed_kernel_config)
                        failed_configs.add(failed_kernel_config)
                    else:
                        print(f"Config {failed_kernel_config} already logged.")
                else:
                    print(f"No kernel configuration found in error message.")
            else:
                # No more OutOfResources error, break the loop
                print(f"{script_path} ran successfully with no errors.")
                break

            # Sleep for a short time before rerunning
            time.sleep(SLEEP_TIME_BETWEEN_RUNS)

if __name__ == "__main__":
    process_scripts()
