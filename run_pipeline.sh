#!/usr/bin/env bash
# set -eu  # Exit on error and undefined vars
set -e  # Exit on error only (remove -u for now)

# Requirements (inside the env):
#   - jupytext
#   - papermill
#   - nbconvert (for Jupyter infrastructure)
#
# Recommended:
#   python -m ipykernel install --user --name=scvi --display-name "Python (scvi)"
# Then set KERNEL_NAME=scvi below.
source /opt/conda/etc/profile.d/conda.sh
conda activate scvi
set -u  # Exit on undefined vars

KERNEL_NAME=${KERNEL_NAME:-scvi}  # set to a registered ipykernel name
TIMEOUT=${TIMEOUT:-96000}          # per-cell timeout seconds (10 hours)

# Parse command line arguments
DATASET_NAME="${1:-${DATASET_NAME:-cite_seq}}"  # Use first arg if provided, else env var, else default



# Function to run a Python script directly (without converting to notebook)
run_script () {
  py_path="$1"
  py_filename="$(basename "${py_path}")"
  py_dir="$(dirname "${py_path}")"

  echo "--------------------------------"
  echo "Running ${py_path} with dataset_name: ${DATASET_NAME}"
  start=$(date +%s)

  # Check if input file exists
  if [ ! -f "${py_path}" ]; then
    echo "ERROR: Input file ${py_path} does not exist!"
    exit 1
  fi

  # Run script directly with dataset_name parameter
  echo "Executing ${py_path}..."
  cd "${py_dir}"

  if [ -n "${DATASET_NAME:-}" ]; then
    echo "Running with dataset_name parameter: ${DATASET_NAME}"
    PYTHONUNBUFFERED=1 python -u "${py_filename}" --dataset_name "${DATASET_NAME}"
  else
    PYTHONUNBUFFERED=1 python -u "${py_filename}"
  fi

  # Check if execution was successful
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to execute ${py_path}!"
    exit 1
  fi

  cd - > /dev/null

  end=$(date +%s)
  echo "Done with ${py_path}"
  echo "Took $((end - start)) seconds"
}

# Function to convert and run as notebook (for compatibility with old workflow)
convert_and_run_streaming () {
  py_path="$1"
  skip_dataset_param="${2:-false}"  # Optional second parameter to skip dataset_name injection
  py_filename="$(basename "${py_path}")"
  base="${py_filename%.py}"

  # Create notebook folder if it doesn't exist
  mkdir -p "${notebook_folder}"

  # Generate timestamp in the format _YYYYMMDD_HHMMSS
  timestamp=$(date +_%Y%m%d_%H%M%S)

  # Save notebook in the notebook_folder with timestamp appended
  nb="${notebook_folder}/${base}${timestamp}.ipynb"
  py_dir="$(dirname "${py_path}")"

  echo "--------------------------------"
  echo "Processing ${py_path}"
  if [ "${skip_dataset_param}" != "true" ]; then
    echo "with dataset_name: ${DATASET_NAME}"
  fi
  echo "Output notebook: ${nb}"
  start=$(date +%s)

  # Check if input file exists
  if [ ! -f "${py_path}" ]; then
    echo "ERROR: Input file ${py_path} does not exist!"
    exit 1
  fi

  # Convert .py -> .ipynb and save to notebook_folder
  echo "Converting ${py_path} to notebook..."
  python -m jupytext --to ipynb "${py_path}" --output "${nb}" --set-kernel "${KERNEL_NAME}"

  # Check if conversion was successful
  if [ ! -f "${nb}" ]; then
    echo "ERROR: Failed to convert ${py_path} to notebook!"
    exit 1
  fi

  # Execute notebook with live output to terminal
  echo "Executing notebook ${nb}..."

  # If dataset_name is set and not skipping, inject it as a parameter
  if [ -n "${DATASET_NAME:-}" ] && [ "${skip_dataset_param}" != "true" ]; then
    echo "Injecting dataset_name parameter: ${DATASET_NAME}"
    papermill "${nb}" "${nb}" \
      -k "${KERNEL_NAME}" \
      --log-output \
      --progress-bar \
      --execution-timeout "${TIMEOUT}" \
      --cwd "${py_dir}" \
      -p dataset_name "${DATASET_NAME}"
  else
    papermill "${nb}" "${nb}" \
      -k "${KERNEL_NAME}" \
      --log-output \
      --progress-bar \
      --execution-timeout "${TIMEOUT}" \
      --cwd "${py_dir}"
  fi

  # Check if execution was successful
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to execute notebook $(realpath "${nb}")!"
    exit 1
  fi

  end=$(date +%s)
  echo "Done with ${py_path}"
  echo "Notebook saved to: $(realpath "${nb}")"
  echo "Took $((end - start)) seconds"
}


# Set dataset name and notebook folder
dataset_name="${DATASET_NAME}"
notebook_folder="notebooks/${dataset_name}"

# Save original plot_flag and set to true (for notebook execution)
CONFIG_FILE="configs/config.json"
ORIGINAL_PLOT_FLAG=$(python -c "import json; f=open('${CONFIG_FILE}'); d=json.load(f); f.close(); print(str(d.get('plot_flag', False)).lower())")
python -c "import json; f=open('${CONFIG_FILE}'); d=json.load(f); f.close(); d['plot_flag']=True; f=open('${CONFIG_FILE}','w'); json.dump(d, f, indent=2); f.close()"

echo "=========================================="
echo "Running ARCADIA Pipeline"
echo "Dataset: ${dataset_name}"
echo "=========================================="

# Step 0: Preprocess dataset
echo ""
echo "=== Step 0: Preprocessing Dataset ==="
preprocess_script="scripts/_0_preprocess_${dataset_name}.py"
if [ ! -f "${preprocess_script}" ]; then
  echo "ERROR: Preprocessing script ${preprocess_script} does not exist!"
  echo "Please check that the dataset name '${dataset_name}' is correct."
  exit 1
fi
convert_and_run_streaming "${preprocess_script}" "true"

# Step 1: Align datasets
echo ""
echo "=== Step 1: Aligning Datasets ==="
convert_and_run_streaming "scripts/_1_align_datasets.py"

# Step 2: Spatial integration
echo ""
echo "=== Step 2: Spatial Integration ==="
convert_and_run_streaming "scripts/_2_spatial_integrate.py"

# Step 3: Generate archetypes
echo ""
echo "=== Step 3: Generating Archetypes ==="
convert_and_run_streaming "scripts/_3_generate_archetypes.py"

# Step 4: Prepare training data
echo ""
echo "=== Step 4: Preparing Training Data ==="
convert_and_run_streaming "scripts/_4_prepare_training.py"

# Clean up scales cache if it exists
[ -f scales_cache.json ] && rm -f scales_cache.json

# Step 5: Train VAE
echo ""
# echo "=== Step 5: Training VAE ==="
run_script "scripts/_5_train_vae.py"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

# Restore original plot_flag
python -c "import json; f=open('${CONFIG_FILE}'); d=json.load(f); f.close(); d['plot_flag']=True if '${ORIGINAL_PLOT_FLAG}' == 'true' else False; f=open('${CONFIG_FILE}','w'); json.dump(d, f, indent=2); f.close()"

# Explicitly exit with success code
exit 0
