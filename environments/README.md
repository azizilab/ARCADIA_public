# Environment Files

This directory contains conda environment specifications for different hardware configurations and Docker setup for containerized execution.

## Conda Environment Files

- `environment_cpu.yaml` - CPU-only environment
- `environment_gpu_cuda11.8.yaml` - GPU environment with CUDA 11.8
- `environment_gpu_cuda12.1.yaml` - GPU environment with CUDA 12.1 (recommended)
- `environment_gpu_cuda12.4.yaml` - GPU environment with CUDA 12.4

### Usage

To create an environment from a YAML file:

```bash
# From the ARCADIA directory
conda env create -f environments/environment_gpu_cuda12.1.yaml
conda activate scvi
pip install -e .
```

### Notes

- All environments use the `scvi` conda environment name
- GPU environments include CUDA toolkit and PyTorch with CUDA support
- CPU environment is for systems without GPU support
- After creating the environment, install the ARCADIA package in editable mode: `pip install -e .`

## Docker Setup

The `docker/` subdirectory contains Docker configuration for containerized execution. Docker provides a reproducible environment that matches the conda setup.

### Quick Start

```bash
# Navigate to docker directory
cd environments/docker

# Build and test
./run_docker.sh test

# Run pipeline
./run_docker.sh pipeline cite_seq

# Interactive session
./run_docker.sh bash
```

### Detailed Documentation

For comprehensive Docker documentation including:
- Prerequisites and installation
- Multiple usage methods (convenience script, manual Docker commands)
- Environment details and features
- Usage examples (pipeline scripts, MLflow, Jupyter)
- Troubleshooting guide
- Advanced usage

See: [`docker/DOCKER_README.md`](docker/DOCKER_README.md)

### Docker Files

- `Dockerfile` - Docker image definition based on conda/miniconda with CUDA support
- `run_docker.sh` - Convenience script to build and run Docker containers
- `DOCKER_README.md` - Comprehensive Docker documentation

### Docker Environment Details

The Docker image includes:
- **Base**: `continuumio/miniconda3:latest`
- **Conda Environment**: `scvi` (created from `environment_gpu_cuda12.1.yaml`)
- **PyTorch**: 2.2.0+cu121 (CUDA 12.1)
- **Python**: 3.10.16
- All dependencies from the conda environment YAML files

### Key Docker Features

1. **Volume Mounting**: Repository mounted as volume for immediate code changes
2. **GPU Support**: Automatic GPU detection and usage
3. **Environment Matching**: Same packages as local conda environment
4. **Development Mode**: Editable package installation
5. **Data Persistence**: All outputs and checkpoints persist across container restarts
