#!/bin/bash

# Define the base directory
base_dir="/home/work/streamk_mi308_tuning/int8"

# List all YAML files under the subdirectories
find "$base_dir" -type f -name "*.yaml" | while read yaml_file; do
  # Extract the directory path where the YAML file is located
  dir_name=$(dirname "$yaml_file")
  
  # Log the directory being processed (stdout)
  echo "Processing directory: $dir_name"
  
  # Change to that directory
  cd "$dir_name" || continue
  
  # Construct the error log filename based on the YAML file name
  yaml_filename=$(basename "$yaml_file")
  error_log="${dir_name}/${yaml_filename%.yaml}_err.info"
  
  # Log the file being processed (stdout)
  echo "Running Tensile for file: $yaml_filename"
  
  # Run the command with the specified device and YAML file, redirect output (stdout and stderr) to the error log
  /home/work/hipBLASLt/tensilelite/Tensile/bin/Tensile --device 7 "$yaml_filename" . > err.info 2>&1
  
  # Log that the command has completed for this file (stdout)
  echo "Completed running Tensile for $yaml_filename. Output saved to $error_log."
  
  # Return to the original directory (if needed)
  cd - || exit
done

# Log that the script has finished processing all directories (stdout)
echo "Finished processing all directories."
