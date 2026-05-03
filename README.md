# Inference Evaluations

This repository contains tools and scripts for evaluating and serving Large Language Models, with a focus on medical benchmarks and quantization techniques.

## 📁 Repository Structure

```
inference-evals/
├── benchmarks/           # Benchmarking and evaluation scripts
│   ├── llm-eval/        # General LLM evaluation using lm-eval-harness
│   └── medical/         # OpenMedicalLLM Leaderboard evaluation
│       ├── open_medical_llm_eval.py    # Main evaluation script
│       ├── download_datasets.py        # Pre-download datasets
│       ├── compare_models.py           # Compare multiple models
│       ├── requirements.txt            # Dependencies
│       └── README.md                   # Medical evaluation docs
├── quantization/        # Model quantization examples
│   └── gptq/           # GPTQ quantization
├── setup/              # Environment setup scripts
│   ├── setup_finetuning_env.sh         # Automated setup script
│   ├── environment.yml                 # Conda environment spec
│   ├── requirements.txt                # Python requirements
│   └── README.md                       # Setup documentation
└── vllm/               # vLLM serving configuration
    └── vllm.service    # Systemd service template
```

## 🚀 Quick Start

### 1. Setup Fine-tuning Environment

```bash
cd setup
chmod +x setup_finetuning_env.sh
./setup_finetuning_env.sh --yes
```

This installs:
- Miniconda (if needed)
- Python 3.12 environment with PyTorch + CUDA 13.0
- Fine-tuning libraries (transformers, peft, trl, accelerate)
- Quantization tools (gptqmodel, bitsandbytes)
- Inference engines (vllm, openai)

### 2. Run Medical Benchmarks

```bash
# Activate environment
source ~/miniconda3/bin/activate finetuning

# Option A: Local vLLM (fastest)
cd benchmarks/medical
python open_medical_llm_eval.py \
    --model_path /path/to/your/model \
    --datasets medqa pubmedqa

# Option B: vLLM Server / OpenAI API
python open_medical_llm_eval.py \
    --api_base http://localhost:8000/v1 \
    --model_name llama-3-8b \
    --datasets medqa pubmedqa
```

### 3. Compare Results

```bash
python compare_models.py --results_dir ./results
```

## 📊 Medical Benchmarks

The `benchmarks/medical` directory includes evaluation on:

| Dataset | Description | Questions |
|---------|-------------|-----------|
| **MedQA (USMLE)** | US Medical Licensing Exam | ~12,700 |
| **PubMedQA** | Biomedical research QA | ~1,000 |
| **MedMCQA** | Indian medical exams | ~194,000 |
| **MMLU** | Medical subjects (Anatomy, Biology, Medicine) | ~700+ |

See [benchmarks/medical/README.md](benchmarks/medical/README.md) for detailed usage.

## 🛠️ Setup & Installation

### Automated Setup

```bash
cd setup
./setup_finetuning_env.sh
```

### Manual Setup

```bash
# Install dependencies
pip install -r setup/requirements.txt

# Or use conda
conda env create -f setup/environment.yml
conda activate finetuning
```

## 🏥 Serving Models

### vLLM Server

Use the provided systemd service template or start manually:

```bash
vllm serve /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

Then evaluate using the API mode:

```bash
python open_medical_llm_eval.py \
    --api_base http://localhost:8000/v1 \
    --model_name your-model-name
```

## 📦 Key Dependencies

| Category | Packages |
|----------|----------|
| **Core** | torch, transformers, accelerate |
| **Fine-tuning** | peft, trl, bitsandbytes |
| **Quantization** | gptqmodel, auto-gptq |
| **Inference** | vllm, openai |
| **Evaluation** | lm-eval, datasets |

## 🔧 Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| **Inference (7B)** | 16 GB GPU | 24 GB GPU |
| **Inference (13B)** | 24 GB GPU | 32 GB GPU |
| **Inference (70B)** | 2x 40 GB GPUs | 4x 40 GB GPUs |
| **QLoRA Fine-tuning (7B)** | 12 GB GPU | 16 GB GPU |
| **QLoRA Fine-tuning (13B)** | 16 GB GPU | 24 GB GPU |

## 📝 License

See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Scripts are executable (`chmod +x`)
- Documentation is updated
- Requirements are specified

## 📚 Additional Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenMedicalLLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard)
