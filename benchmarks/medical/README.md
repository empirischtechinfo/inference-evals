# OpenMedicalLLM Leaderboard Evaluation

This script evaluates Large Language Models on medical question-answering benchmarks using either:
1. **Local vLLM** - Load models directly for fast batched inference
2. **OpenAI-compatible API** - Connect to vLLM server, OpenAI API, or any compatible endpoint

## Supported Datasets

The following datasets from the [OpenMedicalLLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) are supported:

| Dataset | Description | Questions |
|---------|-------------|-----------|
| **MedQA (USMLE)** | US Medical Licensing Exam questions | ~12,700 |
| **PubMedQA** | Biomedical research questions from PubMed | ~1,000 |
| **MedMCQA** | Indian medical entrance exam questions | ~194,000 |
| **MMLU Anatomy** | College-level anatomy questions | ~135 |
| **MMLU Professional Medicine** | Professional medicine questions | ~272 |
| **MMLU College Medicine** | College-level medicine questions | ~173 |
| **MMLU College Biology** | College-level biology questions | ~144 |

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Mode 1: Local vLLM (Fastest for Large Evaluations)

Load models directly and run batched inference locally.

```bash
python open_medical_llm_eval.py \
    --model_path /path/to/your/model \
    --output_dir ./results
```

**Multi-GPU Setup:**
```bash
python open_medical_llm_eval.py \
    --model_path /path/to/large/model \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.85 \
    --batch_size 64
```

**Quantized Models:**
```bash
python open_medical_llm_eval.py \
    --model_path /path/to/gptq-model \
    --quantization gptq \
    --dtype half
```

---

### Mode 2: OpenAI-Compatible API (vLLM Server / OpenAI)

Connect to a running vLLM server or any OpenAI-compatible API endpoint.

#### 2A: vLLM Server (Local or Remote)

First, start a vLLM server:
```bash
vllm serve /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key your-api-key
```

Then run evaluation:
```bash
python open_medical_llm_eval.py \
    --api_base http://localhost:8000/v1 \
    --model_name llama-3-8b \
    --api_key your-api-key \
    --datasets medqa pubmedqa
```

#### 2B: OpenAI API

```bash
export OPENAI_API_KEY="your-api-key"

python open_medical_llm_eval.py \
    --api_base https://api.openai.com/v1 \
    --model_name gpt-4 \
    --datasets medqa pubmedqa mmlu_anatomy \
    --batch_size 10 \
    --max_tokens 50
```

#### 2C: Azure OpenAI

```bash
export OPENAI_API_KEY="your-azure-api-key"

python open_medical_llm_eval.py \
    --api_base https://your-resource.openai.azure.com/openai/deployments/your-deployment/v1 \
    --model_name your-deployment-name \
    --datasets medqa
```

---

### Evaluate on Specific Datasets

```bash
python open_medical_llm_eval.py \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --datasets medqa pubmedqa mmlu_anatomy \
    --output_dir ./results
```

### All Available Options

```bash
python open_medical_llm_eval.py --help
```

## Command-Line Arguments

### Model Source (Choose One)

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to local model (for local vLLM mode) |
| `--api_base` | OpenAI-compatible API base URL (e.g., `http://localhost:8000/v1`) |

### API Mode Options (required with `--api_base`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Model name for API calls (e.g., `gpt-4`, `llama-3-8b`) | Required |
| `--api_key` | API key (or set `OPENAI_API_KEY` env var) | `None` |
| `--temperature` | Sampling temperature | `0.0` |
| `--max_tokens` | Maximum tokens to generate | `32` |
| `--max_retries` | Max retries for failed API calls | `3` |
| `--retry_delay` | Delay between retries (seconds) | `1.0` |

### Local vLLM Options (used with `--model_path`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--tensor_parallel_size` | Number of GPUs for tensor parallelism | `1` |
| `--gpu_memory_utilization` | GPU memory usage (0.0-1.0) | `0.9` |
| `--max_model_len` | Maximum context length | Model default |
| `--dtype` | Data type (auto/half/bfloat16/float) | `auto` |
| `--quantization` | Quantization method (awq/gptq/squeezellm) | None |

### Common Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--datasets` | Datasets to evaluate on | All datasets |
| `--output_dir` | Directory to save results | `./results` |
| `--cache_dir` | Directory to cache datasets | `./datasets_cache` |
| `--batch_size` | Inference batch size | `32` |
| `--save_predictions` | Save detailed predictions JSON | False |

## Example Output

```
============================================================
📋 RESULTS FOR: Llama-2-7b-chat-hf
============================================================

| Dataset                        | Correct | Total | Accuracy |
|---------|---------|-------|----------|
| MedQA (USMLE)                  |    3500 |  12723 |   27.51% |
| PubMedQA                       |     450 |   1000 |   45.00% |
| MedMCQA                        |   25000 |  65000 |   38.46% |
| MMLU Anatomy                   |      75 |    135 |   55.56% |
| MMLU Professional Medicine     |     150 |    272 |   55.15% |
|---------|---------|-------|----------|
| OVERALL                        |   29175 |  79130 |   36.87% |

💾 Results saved to: ./results/medical_eval_Llama-2-7b-chat-hf.json
💾 CSV summary saved to: ./results/medical_eval_Llama-2-7b-chat-hf.csv
```

## Comparing Models

The generated CSV files can be easily combined for comparison:

```python
import pandas as pd
import glob

# Load all result CSVs
csv_files = glob.glob("./results/*.csv")
dfs = [pd.read_csv(f) for f in csv_files]

# Combine and compare
combined = pd.concat(dfs, keys=[f.split('/')[-1].replace('.csv', '') for f in csv_files])
print(combined.pivot_table(values='Accuracy', index='Dataset', columns='Model'))
```

Or use the included comparison script:
```bash
python compare_models.py --results_dir ./results
```

## Mode Comparison

| Feature | Local vLLM | OpenAI API |
|---------|-----------|------------|
| **Speed** | Fastest (batched) | Slower (sequential) |
| **Setup** | Requires GPU, model download | Just API key |
| **Cost** | Free (after download) | Per-token pricing |
| **Best For** | Large evaluations, many models | Quick tests, proprietary models |

## Notes

- First run will download datasets from HuggingFace (cached for subsequent runs)
- Local vLLM requires a CUDA-capable GPU
- Adjust `--batch_size` based on your GPU memory (local) or rate limits (API)
- For models with limited context, set `--max_model_len` appropriately

## Troubleshooting

### Local vLLM Mode

**Out of Memory**
- Reduce `--batch_size`
- Reduce `--gpu_memory_utilization`
- Enable quantization with `--quantization`
- Use `--tensor_parallel_size` to spread across GPUs

### API Mode

**Rate Limiting**
- Reduce `--batch_size` (parallel requests)
- Increase `--retry_delay`
- Contact provider for higher limits

**Connection Errors**
- Verify `--api_base` URL is correct
- Check if API key is set correctly
- Ensure vLLM server is running and accessible

### Dataset Loading Issues
- Ensure you have `trust_remote_code=True` acceptance for datasets
- Check your HuggingFace token if accessing gated models
- Run `python download_datasets.py` first to pre-cache

### Slow Performance
- **Local**: Increase `--batch_size` if you have memory available
- **API**: Check provider's rate limits and parallel request allowances
- Enable `HF_HUB_ENABLE_HF_TRANSFER=1` for faster dataset downloads
