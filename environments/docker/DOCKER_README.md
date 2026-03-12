# Docker Setup for ARCADIA

This directory contains Docker configuration files to run the ARCADIA pipeline in a containerized environment. Docker provides a reproducible, isolated environment that ensures consistent execution across different systems.

## Files

- `Dockerfile` - Docker image definition based on conda/miniconda with CUDA support
- `run_docker.sh` - Convenience script to build and run Docker containers
- `.dockerignore` - Files to exclude from Docker build context (located in project root)

## Prerequisites

### Required
- **Docker**: Docker Engine 20.10+ installed and running
  - Installation: [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux)
  - Verify: `docker --version`

### Optional but Recommended
- **NVIDIA Docker runtime**: For GPU support
  - Installation: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - Verify: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### System Requirements
- **Disk Space**: ~10GB for the Docker image
- **Memory**: Minimum 16GB RAM (32GB recommended)
- **GPU**: NVIDIA GPU with CUDA 12.1 support (optional, for faster training)

## Quick Start

### Method 1: Using the Convenience Script (Recommended)

The `run_docker.sh` script simplifies building and running Docker containers:

```bash
# From the ARCADIA project root directory
cd ARCADIA/environments/docker

# Build and test the Docker image
./run_docker.sh test

# Run the pipeline with a specific dataset
./run_docker.sh pipeline cite_seq
./run_docker.sh pipeline tonsil

# Start an interactive bash session
./run_docker.sh bash
```

### Method 2: Manual Docker Commands

For more control, use Docker commands directly:

```bash
# Build the image (from ARCADIA directory)
cd ARCADIA
docker build -f environments/docker/Dockerfile -t arcadia:latest .

# Run the pipeline
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  -e CONDA_DEFAULT_ENV=scvi \
  -e PATH=/opt/conda/envs/scvi/bin:$PATH \
  arcadia:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate scvi && \
           pip install -e . && \
           bash run_pipeline_notebooks.sh cite_seq"

# Interactive session
docker run --rm -it --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  -e CONDA_DEFAULT_ENV=scvi \
  -e PATH=/opt/conda/envs/scvi/bin:$PATH \
  arcadia:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate scvi && \
           pip install -e . && \
           /bin/bash"
```

## Environment Details

The Docker image includes:

- **Base Image**: `continuumio/miniconda3:latest`
- **Conda Environment**: `scvi` (created from `environments/environment_gpu_cuda12.1.yaml`)
- **Python**: 3.10.16
- **PyTorch**: 2.2.0+cu121 (CUDA 12.1)
- **CUDA**: 12.1 support
- **Key Packages**:
  - scvi-tools (latest)
  - scanpy, anndata
  - pytorch-lightning, torchmetrics
  - jupyter, jupytext, papermill (for notebook execution)
  - numpy, pandas, scipy, scikit-learn
  - matplotlib, seaborn (for visualization)

## Key Features

1. **Volume Mounting**: The repository is mounted (not copied) so changes are reflected immediately
   - Changes to code are immediately available in the container
   - No need to rebuild the image for code changes
   - Data persistence across container restarts

2. **GPU Support**: Uses `--gpus all` for NVIDIA GPU access
   - Automatically detects and uses available GPUs
   - Falls back to CPU if GPU is unavailable (slower but functional)

3. **Environment Matching**: Matches your local `scvi` conda environment
   - Same package versions as local development
   - Consistent behavior across environments

4. **Development Mode**: Installs `arcadia` package in editable mode
   - Code changes are immediately reflected
   - No need to reinstall after modifications

5. **Data Persistence**:
   - Repository directory is mounted as `/workspace`
   - Data files in `CODEX_RNA_seq/data/` are accessible
   - MLflow runs stored in `mlruns/` directory persist
   - All outputs and checkpoints are saved to mounted volumes

## Usage Examples

### Running Different Pipeline Scripts

```bash
# Run the simple pipeline (no plots) - using run_docker.sh
cd ARCADIA/environments/docker
./run_docker.sh pipeline cite_seq

# Run the notebook-based pipeline (with plots) - requires manual docker command
# since run_docker.sh uses run_pipeline_direct.sh
docker run --rm -i --gpus all \
  -v $(pwd)/../..:/workspace \
  -w /workspace \
  -e CONDA_DEFAULT_ENV=scvi \
  -e PATH=/opt/conda/envs/scvi/bin:$PATH \
  arcadia:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate scvi && \
           cd /workspace/ARCADIA && pip install -e . > /dev/null 2>&1 && \
           cd /workspace && bash run_pipeline_notebooks.sh tonsil"
```

### Running Individual Pipeline Steps

```bash
# Run a specific script - using interactive session
cd ARCADIA/environments/docker
./run_docker.sh bash
# Then inside container:
# python scripts/_1_align_datasets.py --dataset_name cite_seq
```

### Accessing MLflow UI

```bash
# Start MLflow UI in the container
docker run --rm -i --gpus all \
  -v $(pwd)/../..:/workspace \
  -w /workspace \
  -p 5000:5000 \
  -e CONDA_DEFAULT_ENV=scvi \
  -e PATH=/opt/conda/envs/scvi/bin:$PATH \
  arcadia:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate scvi && \
           cd /workspace/ARCADIA && pip install -e . > /dev/null 2>&1 && \
           mlflow ui --host 0.0.0.0 --port 5000"
```

Then access MLflow at `http://localhost:5000` from your host machine.

### Running Jupyter Notebooks

```bash
# Start Jupyter server
docker run --rm -i --gpus all \
  -v $(pwd)/../..:/workspace \
  -w /workspace \
  -p 8888:8888 \
  -e CONDA_DEFAULT_ENV=scvi \
  -e PATH=/opt/conda/envs/scvi/bin:$PATH \
  arcadia:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate scvi && \
           cd /workspace/ARCADIA && pip install -e . > /dev/null 2>&1 && \
           jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

Access Jupyter at `http://localhost:8888` from your host machine.

## Advanced Usage

### Custom Environment Variables

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/../..:/workspace \
  -w /workspace \
  -e CONDA_DEFAULT_ENV=scvi \
  -e PATH=/opt/conda/envs/scvi/bin:$PATH \
  -e CUSTOM_VAR=value \
  arcadia:latest \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate scvi && \
           cd /workspace/ARCADIA && pip install -e . > /dev/null 2>&1 && \
           /bin/bash"
```

### Mount Additional Volumes

Add additional volume mounts to the docker run command:

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/../..:/workspace \
  -v /path/to/data:/data \
  -w /workspace \
  -e CONDA_DEFAULT_ENV=scvi \
  -e PATH=/opt/conda/envs/scvi/bin:$PATH \
  arcadia:latest bash
```

### Build with Different Base Image

Edit `Dockerfile` to use a different base image or CUDA version.

### Multi-stage Builds

For smaller images, consider multi-stage builds (advanced Docker feature).
