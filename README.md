

```markdown
# Recursive Self-Refinement Retrieval

A multi-modal retrieval project with recursive self-refinement using BLIP and reinforcement learning.

## Setup

### 1. Create Environment
```bash
conda create -n recursive_retrieval python=3.8
conda activate recursive_retrieval
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API (if needed)
```bash
export OPENAI_API_KEY="your-key"
```

## Usage

### Training
```bash
python main.py --mode train --dataset flickr30k
```

### Evaluation
```bash
python main.py --mode eval --dataset flickr30k
```

### Interactive Demo
```bash
python inference/demo.py
```

## Project Structure

- `data/`: Data loading and preprocessing.
  - `datasets/`: Specific dataset implementations (e.g., `infoseek.py`, `flickr30k.py`, `vqa.py`).
  - `dataloader.py`: Data loader.
  - `preprocessing.py`: Data preprocessing tools.
- `models/`: Model-related files.
  - `base_retriever.py`: BLIP-based retrieval model.
  - `uncertainty.py`: Uncertainty estimation module.
  - `recursive_model.py`: Recursive self-refinement model.
- `agents/`: Agent-related files.
  - `meta_controller.py`: Meta-controller implementation.
  - `environment.py`: RL environment definition.
  - `reward.py`: Reward function definition.
  - `rl/`: Reinforcement learning algorithms.
    - `ppo.py`: PPO algorithm implementation.
    - `buffer.py`: Experience replay buffer.
    - `utils.py`: RL utility functions.
- `training/`: Training-related scripts.
  - `train_retriever.py`: Retriever model training script.
  - `train_agent.py`: Agent training script.
  - `evaluate.py`: Evaluation script.
- `utils/`: Utility functions.
  - `metrics.py`: Evaluation metrics.
  - `visualization.py`: Visualization tools.
  - `logger.py`: Logging utilities.
  - `helpers.py`: General helper functions.
- `inference/`: Inference-related files.
  - `predictor.py`: Prediction implementation.
  - `demo.py`: Interactive demo script.
- `config/`: Configuration files.
  - `default_config.yaml`: Default configuration parameters.
  - `model_config.yaml`: Model-specific parameters.
  - `agent_config.yaml`: Agent-specific parameters.
  - `dataset_configs/`: Dataset-specific configurations.
- `main.py`: Main program entry point.

## Requirements
See `requirements.txt` for dependencies:
```
torch==2.2.0
transformers==4.37.2
pillow==10.2.0
numpy==1.26.4
tqdm==4.66.2
scikit-learn==1.4.2
```

## Notes
- Ensure CUDA is compatible with PyTorch 2.2.0 (e.g., CUDA 11.8) on Ubuntu 22.04.
- Replace `your-key` with a valid API key if required.
```

