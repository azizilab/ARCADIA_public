#!/bin/bash
# Script to build and run ARCADIA pipeline in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ARCADIA Docker Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
    echo -e "${GREEN}✓ GPU detected (nvidia-smi available)${NC}"
else
    echo -e "${YELLOW}⚠ GPU not detected, will use CPU-only mode${NC}"
fi

# Build Docker image
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Determine if we're running from docker folder or ARCADIA root
if [[ "$SCRIPT_DIR" == *"/environments/docker" ]]; then
    # Running from docker folder - go up to ARCADIA root
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile"
else
    # Running from ARCADIA root
    PROJECT_ROOT="$SCRIPT_DIR"
    DOCKERFILE_PATH="environments/docker/Dockerfile"
fi
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

# Check if --no-cache flag is provided or if requirements.txt was recently modified
FORCE_REBUILD=false
if [[ "$*" == *"--no-cache"* ]] || [[ "$*" == *"--rebuild"* ]]; then
    FORCE_REBUILD=true
fi

# Check if image exists
IMAGE_EXISTS=false
if docker images | grep -q "arcadia.*latest"; then
    IMAGE_EXISTS=true
fi

# Build from PROJECT_ROOT (ARCADIA) directory only if needed
if [ "$FORCE_REBUILD" = true ] || [ "$IMAGE_EXISTS" = false ]; then
    if [ "$FORCE_REBUILD" = true ]; then
        echo -e "\n${YELLOW}Force rebuilding Docker image (no cache)...${NC}"
    else
        echo -e "\n${GREEN}Building Docker image...${NC}"
    fi

    cd "$PROJECT_ROOT" || exit 1
    if [ "$FORCE_REBUILD" = true ]; then
        docker build --no-cache -f "$DOCKERFILE_PATH" -t arcadia:latest .
    else
        docker build -f "$DOCKERFILE_PATH" -t arcadia:latest .
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Docker build failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Docker image built successfully${NC}"
    cd "$REPO_ROOT" || exit 1
else
    echo -e "\n${GREEN}✓ Docker image already exists, skipping build${NC}"
fi

# Run Docker container
echo -e "\n${GREEN}Starting Docker container...${NC}"

# Prepare docker run command
# Use -it for interactive, -i for non-interactive
if [ "$1" = "bash" ]; then
    DOCKER_CMD="docker run --rm -it"
else
    DOCKER_CMD="docker run --rm -i"
fi

# Add GPU support if available
if [ "$GPU_AVAILABLE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD --gpus all"
fi

# Mount the repository root (parent of ARCADIA) so we can access run_pipeline_direct.sh
# Always mount REPO_ROOT regardless of where script is run from
DOCKER_CMD="$DOCKER_CMD -v $REPO_ROOT:/workspace"

# Set working directory
DOCKER_CMD="$DOCKER_CMD -w /workspace"

# Set environment variables
DOCKER_CMD="$DOCKER_CMD -e CONDA_DEFAULT_ENV=scvi"
DOCKER_CMD="$DOCKER_CMD -e PATH=/opt/conda/envs/scvi/bin:\$PATH"

# Image name
DOCKER_CMD="$DOCKER_CMD arcadia:latest"

# Command to run
if [ "$1" = "bash" ]; then
    echo -e "${GREEN}Starting interactive bash session...${NC}"
    eval "$DOCKER_CMD bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate scvi && cd /workspace/ARCADIA && pip install -e . > /dev/null 2>&1 && cd /workspace && /bin/bash'"
elif [ "$1" = "pipeline" ]; then
    DATASET_NAME=${2:-cite_seq}
    echo -e "${GREEN}Running pipeline with dataset: $DATASET_NAME${NC}"
    eval "$DOCKER_CMD bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate scvi && cd /workspace/ARCADIA && pip install -e . > /dev/null 2>&1 && cd /workspace && bash run_pipeline_direct.sh $DATASET_NAME'"
elif [ "$1" = "test" ]; then
    echo -e "${GREEN}Testing arcadia package import...${NC}"
    eval "$DOCKER_CMD bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate scvi && cd /workspace/ARCADIA && pip install -e . > /dev/null 2>&1 && python -c \"import arcadia; print(\\\"✓ arcadia imported successfully\\\")\"'"
else
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 bash       - Start interactive bash session"
    echo -e "  $0 pipeline [dataset_name]   - Run the ARCADIA pipeline (default: cite_seq)"
    echo -e "  $0 test       - Test arcadia package import"
    echo -e ""
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  --no-cache    - Force rebuild Docker image without cache (use after changing requirements.txt)"
    echo -e "  --rebuild     - Same as --no-cache"
    echo -e ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0 bash"
    echo -e "  $0 test"
    echo -e "  $0 pipeline cite_seq"
    echo -e "  $0 --no-cache bash       # Force rebuild before starting bash"
    echo -e "  $0 --rebuild pipeline    # Force rebuild before running pipeline"
    exit 1
fi
