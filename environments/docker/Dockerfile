# Dockerfile for ARCADIA pipeline
# Based on conda/miniconda with CUDA support for GPU

FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file and requirements.txt
COPY environments/environment_gpu_cuda12.1.yaml /tmp/environment.yaml
COPY environments/requirements.txt /tmp/requirements.txt

# Create a modified environment file without torch packages in pip section
# (they will be installed separately from PyTorch index)
RUN sed -i '/^      - torch==/d; /^      - torchaudio==/d; /^      - torchvision==/d' /tmp/environment.yaml

# Create conda environment from modified YAML file
RUN conda env create -f /tmp/environment.yaml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "scvi", "/bin/bash", "-c"]

# Install correct PyTorch versions matching scvi environment from PyTorch index
# torch==2.2.0+cu121, torchaudio==2.2.0+cu121, torchvision==0.17.0+cu121
RUN conda run -n scvi pip install --no-cache-dir \
    torch==2.2.0+cu121 \
    torchaudio==2.2.0+cu121 \
    torchvision==0.17.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install dependencies from requirements.txt
# Note: pytorch-lightning and torchmetrics are included in requirements.txt
# Note: Some packages may already be installed via conda environment, but pip will skip them
RUN conda run -n scvi pip install --no-cache-dir -r /tmp/requirements.txt || true

# Explicitly install kneed to ensure it's available
RUN conda run -n scvi pip install --no-cache-dir kneed==0.8.5

# Copy ARCADIA setup script for patching scVI library
COPY src/arcadia/utils/setup_scvi_patch.py /tmp/setup_scvi_patch.py

# Patch scVI library to support custom training plans
# This patches scvi-tools==1.2.2.post2 to add 'self._training_plan = training_plan' at line 131
# The patch script is run directly without needing the full ARCADIA package installed
RUN conda run -n scvi python /tmp/setup_scvi_patch.py

# Set environment variables
ENV CONDA_DEFAULT_ENV=scvi
ENV PATH=/opt/conda/envs/scvi/bin:$PATH

# Note: The ARCADIA package will be installed/reinstalled at runtime via volume mount in run_docker.sh
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate scvi" >> ~/.bashrc

# Set the default command
CMD ["/bin/bash"]
