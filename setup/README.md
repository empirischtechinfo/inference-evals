# Fine-tuning Environment Setup

This directory contains scripts and configurations for setting up a complete fine-tuning environment for Large Language Models.

## Quick Start

### Option 1: Automated Setup Script (Recommended)

```bash
# Navigate to setup directory
cd setup

# Make script executable
chmod +x setup_finetuning_env.sh

# Run setup (interactive mode)
./setup_finetuning_env.sh

# Or run with auto-yes (non-interactive)
./setup_finetuning_env.sh --yes

# Skip system updates (useful if already updated)
./setup_finetuning_env.sh --yes --skip-system-update

# Use custom environment name
./setup_finetuning_env.sh --env-name my_finetuning_env
```

### Option 2: Conda Environment File

```bash
# Using the environment.yml file
conda env create -f environment.yml

# Activate the environment
conda activate finetuning

# Set required environment variable
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### Option 3: Manual Step-by-Step

See the [Manual Setup](#manual-setup) section below.

## What Gets Installed

### System Dependencies
- Build essentials (build-essential, cmake, ninja-build)
- GCC 12 (g++-12)
- Miniconda (if not present)

### Python Environment
- Python 3.12
- PyTorch with CUDA 13.0 support

### Core Fine-tuning Libraries
| Package | Purpose |
|---------|---------|
| **transformers** | Hugging Face model library |
| **peft** | Parameter Efficient Fine-Tuning (LoRA, QLoRA, etc.) |
| **trl** | Transformer Reinforcement Learning (PPO, DPO, etc.) |
| **accelerate** | Multi-GPU and mixed precision training |
| **bitsandbytes** | 8-bit and 4-bit quantization |
| **gptqmodel** | GPTQ quantization support |

### Inference & Evaluation
| Package | Purpose |
|---------|---------|
| **vllm** | Fast inference serving |
| **openai** | OpenAI-compatible API client |
| **lm-eval** | Language model evaluation harness |
| **datasets** | Hugging Face datasets library |

### Utilities
| Package | Purpose |
|---------|---------|
| **wandb** | Weights & Biases experiment tracking |
| **tensorboard** | Training visualization |
| **python-dotenv** | Environment variable management |
| **langdetect** | Language detection |

## Usage After Setup

### Activate Environment

```bash
# Standard activation
source ~/miniconda3/bin/activate finetuning
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Or use the convenience script
./activate_finetuning.sh
```

### Verify Installation

```bash
python verify_installation.py
```

Expected output:
```
============================================================
Fine-tuning Environment Verification
============================================================
Python version: 3.12.x

PyTorch version: 2.3.x+cu130
CUDA available: True
CUDA version: 13.0
GPU count: 1
  GPU 0: NVIDIA RTX 4090

Core Libraries:
✓ transformers
✓ peft
✓ trl
✓ accelerate
✓ bitsandbytes

============================================================
✓ All core libraries installed successfully!
============================================================
```

## Manual Setup

If you prefer to set up manually or need to customize:

### 1. System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y g++-12

# Set LD_PRELOAD
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
```

### 3. Create Environment

```bash
conda create -n finetuning python=3.12 -y
conda activate finetuning
```

### 4. Install PyTorch with CUDA 13.0

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 5. Install Fine-tuning Libraries

```bash
pip install transformers accelerate bitsandbytes trl peft python-dotenv
```

### 6. Install Optional Packages

```bash
# GPTQ quantization
pip install -v gptqmodel --no-build-isolation

# Evaluation
pip install "lm-eval>=0.4.7"

# Inference
pip install vllm openai

# Additional utilities
pip install langdetect tqdm datasets
```

### 7. Install from requirements.txt

```bash
pip install -r requirements.txt
```

## Troubleshooting

### CUDA/GPU Issues

**Problem:** `CUDA out of memory` during training

**Solution:**
```bash
# Enable memory-efficient attention
pip install xformers

# Or use DeepSpeed for distributed training
pip install deepspeed
```

### GCC/Compilation Issues

**Problem:** `gcc: error: unrecognized command-line option`

**Solution:**
```bash
# Ensure GCC 12 is being used
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# Reinstall the problematic package
pip install --force-reinstall --no-cache-dir gptqmodel --no-build-isolation
```

### Conda Not Found

**Problem:** `conda: command not found` after installation

**Solution:**
```bash
# Reinitialize conda for bash
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### libstdc++ Errors

**Problem:** `version GLIBCXX_3.4.32 not found`

**Solution:**
```bash
# Ensure LD_PRELOAD is set
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Add to .bashrc for persistence
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' >> ~/.bashrc
```

## Fine-tuning Example

After setup, here's a minimal fine-tuning example:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Setup trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    save_steps=100,
    logging_steps=10,
)

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Additional Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [vLLM Documentation](https://docs.vllm.ai/)

## Files in This Directory

| File | Purpose |
|------|---------|
| `setup_finetuning_env.sh` | Main automated setup script |
| `requirements.txt` | Python package requirements |
| `environment.yml` | Conda environment specification |
| `README.md` | This documentation file |

## Contributing

To update the setup script or add new packages:

1. Edit `setup_finetuning_env.sh` for system-level changes
2. Edit `requirements.txt` for Python packages
3. Edit `environment.yml` for conda-specific setup
4. Test on a fresh system if possible
