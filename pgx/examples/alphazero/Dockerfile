FROM nvcr.io/nvidia/jax:24.10-py3

ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create checkpoints directory
RUN mkdir -p checkpoints

# Set environment variables for optimal GPU performance
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
