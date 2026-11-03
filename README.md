# Training LLM

## Installation

### Requirements
- Python 3.15.2
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/training-llm.git
cd training-llm

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.15.2
source .venv/bin/activate

# Install dependencies via uv
uv sync

# Install pre-commit
pre-commit install
```

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

## How To Use

To run the training script, use the following command:

```bash
uv run train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

- `CONFIG_NAME`: A config file from `src/configs`
- `HYDRA_CONFIG_ARGUMENTS`: Optional Hydra overrides
