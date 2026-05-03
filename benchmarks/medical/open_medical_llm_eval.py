"""
OpenMedicalLLM Leaderboard Benchmark Script

This script downloads the medical datasets used in the OpenMedicalLLM Leaderboard
and evaluates models using either:
1. vLLM (local model loading) for fast inference
2. OpenAI-compatible API (vLLM server, OpenAI, etc.)

Datasets:
- MedQA (USMLE): Medical licensing exam questions
- PubMedQA: Biomedical research questions
- MedMCQA: Indian medical exam questions
- MMLU (Professional Medicine, Anatomy, etc.)

Usage:
    # Local vLLM
    python open_medical_llm_eval.py --model_path /path/to/model --output_dir results/

    # OpenAI-compatible API (vLLM server)
    python open_medical_llm_eval.py --api_base http://localhost:8000/v1 --model_name llama-3-8b

    # OpenAI API
    python open_medical_llm_eval.py --api_base https://api.openai.com/v1 --model_name gpt-4 --api_key $OPENAI_API_KEY
"""

import argparse
import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Configuration for a medical dataset."""
    name: str
    dataset_path: str
    config: Optional[str]
    question_field: str
    choices_field: Optional[str]
    answer_field: str
    answer_format: str  # 'letter' (A/B/C/D), 'index', or 'text'
    split: str
    choices_labels: Optional[List[str]] = None
    extract_choices_from_options: bool = False


# Dataset configurations for OpenMedicalLLM Leaderboard
DATASET_CONFIGS = {
    "medqa": DatasetConfig(
        name="MedQA (USMLE)",
        dataset_path="bigbio/med_qa",
        config="med_qa_en",
        question_field="question",
        choices_field="options",
        answer_field="answer_idx",
        answer_format="index",
        split="test",
        choices_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    ),
    "pubmedqa": DatasetConfig(
        name="PubMedQA",
        dataset_path="pubmedqa",
        config="pqa_artificial",
        question_field="question",
        choices_field=None,
        answer_field="final_decision",
        answer_format="text",
        split="train",
        choices_labels=["yes", "no", "maybe"],
    ),
    "medmcqa": DatasetConfig(
        name="MedMCQA",
        dataset_path="medmcqa",
        config=None,
        question_field="question",
        choices_field=None,
        answer_field="cop",  # correct option
        answer_format="index",
        split="test",
        choices_labels=["A", "B", "C", "D"],
        extract_choices_from_options=True,
    ),
    "mmlu_anatomy": DatasetConfig(
        name="MMLU Anatomy",
        dataset_path="cais/mmlu",
        config="anatomy",
        question_field="question",
        choices_field="choices",
        answer_field="answer",
        answer_format="index",
        split="test",
        choices_labels=["A", "B", "C", "D"],
    ),
    "mmlu_professional_medicine": DatasetConfig(
        name="MMLU Professional Medicine",
        dataset_path="cais/mmlu",
        config="professional_medicine",
        question_field="question",
        choices_field="choices",
        answer_field="answer",
        answer_format="index",
        split="test",
        choices_labels=["A", "B", "C", "D"],
    ),
    "mmlu_college_medicine": DatasetConfig(
        name="MMLU College Medicine",
        dataset_path="cais/mmlu",
        config="college_medicine",
        question_field="question",
        choices_field="choices",
        answer_field="answer",
        answer_format="index",
        split="test",
        choices_labels=["A", "B", "C", "D"],
    ),
    "mmlu_college_biology": DatasetConfig(
        name="MMLU College Biology",
        dataset_path="cais/mmlu",
        config="college_biology",
        question_field="question",
        choices_field="choices",
        answer_field="answer",
        answer_format="index",
        split="test",
        choices_labels=["A", "B", "C", "D"],
    ),
}


class MedicalDatasetLoader:
    """Loads and prepares medical datasets for evaluation."""

    def __init__(self, cache_dir: str = "./datasets_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load_dataset(self, config: DatasetConfig) -> List[Dict[str, Any]]:
        """Load a dataset based on its configuration."""
        print(f"\n📥 Downloading {config.name}...")

        try:
            if config.config:
                dataset = load_dataset(
                    config.dataset_path,
                    config.config,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    config.dataset_path,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )

            split_data = dataset[config.split]
            examples = []

            for item in split_data:
                example = self._format_example(item, config)
                if example:
                    examples.append(example)

            print(f"   ✓ Loaded {len(examples)} examples from {config.name}")
            return examples

        except Exception as e:
            print(f"   ✗ Error loading {config.name}: {e}")
            return []

    def _format_example(self, item: Dict, config: DatasetConfig) -> Optional[Dict]:
        """Format a dataset item into a standardized example."""
        try:
            question = item.get(config.question_field, "")

            # Handle choices extraction
            choices = []
            if config.extract_choices_from_options and config.name == "MedMCQA":
                # MedMCQA has separate fields for each option
                for i, label in enumerate(config.choices_labels or ["A", "B", "C", "D"]):
                    opt_key = f"opa" if i == 0 else f"op{chr(ord('a') + i)}"
                    if opt_key in item:
                        choices.append(item[opt_key])
            elif config.choices_field:
                choices_data = item.get(config.choices_field, [])
                if isinstance(choices_data, dict):
                    # Handle dict format (e.g., MedQA)
                    choices = [choices_data.get(k, "") for k in sorted(choices_data.keys())]
                elif isinstance(choices_data, list):
                    choices = choices_data
                else:
                    choices = []
            elif config.name == "PubMedQA":
                choices = ["yes", "no", "maybe"]

            # Get the correct answer
            correct_answer = item.get(config.answer_field)
            if config.answer_format == "index" and isinstance(correct_answer, int):
                correct_label = config.choices_labels[correct_answer] if correct_answer < len(config.choices_labels) else str(correct_answer)
            elif config.answer_format == "letter":
                correct_label = correct_answer.upper() if isinstance(correct_answer, str) else config.choices_labels[correct_answer]
            else:
                correct_label = str(correct_answer).lower() if isinstance(correct_answer, str) else str(correct_answer)

            return {
                "question": question,
                "choices": choices,
                "correct_answer": correct_label,
                "raw_answer": correct_answer,
            }
        except Exception as e:
            print(f"   Warning: Failed to format example: {e}")
            return None


class MedicalPromptBuilder:
    """Builds prompts for medical multiple-choice questions."""

    SYSTEM_PROMPT = (
        "You are a medical expert answering multiple-choice questions. "
        "Answer with only the letter of the correct choice (A, B, C, D, etc.) or the exact answer. "
        "Be concise and respond with just the answer."
    )

    @classmethod
    def build_prompt(cls, question: str, choices: List[str], choices_labels: List[str]) -> str:
        """Build a standardized multiple-choice prompt."""
        prompt_parts = [question, "\n"]

        for i, choice in enumerate(choices):
            if i < len(choices_labels):
                label = choices_labels[i]
                prompt_parts.append(f"{label}. {choice}")

        prompt_parts.append("\nAnswer:")
        return "\n".join(prompt_parts)

    @classmethod
    def build_chat_messages(cls, question: str, choices: List[str], choices_labels: List[str]) -> List[Dict[str, str]]:
        """Build chat messages for instruction-tuned models."""
        user_prompt = cls.build_prompt(question, choices, choices_labels)
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]


class AnswerExtractor:
    """Extracts answers from model outputs."""

    @staticmethod
    def extract_letter_answer(text: str, valid_options: List[str]) -> Optional[str]:
        """Extract a letter answer (A, B, C, D) from text."""
        text = text.strip().upper()

        # Direct match at start
        if text and text[0] in valid_options:
            return text[0]

        # Pattern: "The answer is X" or "Answer: X"
        patterns = [
            r"ANSWER[:\s]+([A-D])",
            r"THE ANSWER IS[:\s]+([A-D])",
            r"\b([A-D])\b",  # Any standalone letter
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def extract_text_answer(text: str, valid_options: List[str]) -> Optional[str]:
        """Extract a text answer (yes/no/maybe) from text."""
        text_lower = text.strip().lower()

        # Direct match
        if text_lower in valid_options:
            return text_lower

        # Check for mentions in text
        for option in valid_options:
            if option in text_lower:
                return option

        # Pattern matching
        if "yes" in text_lower:
            return "yes"
        elif "no" in text_lower:
            return "no"
        elif "maybe" in text_lower or "perhaps" in text_lower or "uncertain" in text_lower:
            return "maybe"

        return None


class BaseMedicalEvaluator(ABC):
    """Abstract base class for medical evaluators."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_responses(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        batch_size: int = 32,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        pass

    def evaluate_dataset(
        self,
        examples: List[Dict[str, Any]],
        config: DatasetConfig,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Evaluate a single dataset."""
        print(f"\n📊 Evaluating on {config.name} ({len(examples)} questions)...")

        correct = 0
        total = len(examples)
        predictions = []

        # Prepare all prompts as chat messages
        all_messages = []
        for example in examples:
            messages = MedicalPromptBuilder.build_chat_messages(
                example["question"],
                example["choices"],
                config.choices_labels or ["A", "B", "C", "D"]
            )
            all_messages.append(messages)

        # Generate responses in batches
        all_outputs = []
        for i in tqdm(range(0, len(all_messages), batch_size), desc="Generating"):
            batch_messages = all_messages[i:i + batch_size]
            outputs = self.generate_responses(batch_messages, batch_size)
            all_outputs.extend(outputs)

        # Extract and compare answers
        for idx, (example, generated_text) in enumerate(zip(examples, all_outputs)):
            generated_text = generated_text.strip()

            # Extract answer based on format
            if config.answer_format == "index" or config.answer_format == "letter":
                predicted = AnswerExtractor.extract_letter_answer(
                    generated_text,
                    config.choices_labels or ["A", "B", "C", "D"]
                )
            else:
                predicted = AnswerExtractor.extract_text_answer(
                    generated_text,
                    config.choices_labels or ["yes", "no", "maybe"]
                )

            is_correct = predicted == example["correct_answer"]
            if is_correct:
                correct += 1

            predictions.append({
                "question": example["question"][:100] + "..." if len(example["question"]) > 100 else example["question"],
                "predicted": predicted,
                "correct": example["correct_answer"],
                "generated": generated_text[:100],
                "is_correct": is_correct,
            })

        accuracy = (correct / total) * 100 if total > 0 else 0

        return {
            "dataset": config.name,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "predictions": predictions,
        }

    def evaluate_all(
        self,
        dataset_names: List[str],
        loader: MedicalDatasetLoader,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Evaluate on all specified datasets."""
        results = {
            "model": self.model_name,
            "datasets": {},
        }

        overall_correct = 0
        overall_total = 0

        for name in dataset_names:
            if name not in DATASET_CONFIGS:
                print(f"⚠ Unknown dataset: {name}, skipping...")
                continue

            config = DATASET_CONFIGS[name]
            examples = loader.load_dataset(config)

            if not examples:
                print(f"⚠ No examples loaded for {name}, skipping...")
                continue

            # Optionally limit for quick testing
            # examples = examples[:100]

            dataset_result = self.evaluate_dataset(examples, config, batch_size)
            results["datasets"][name] = dataset_result

            overall_correct += dataset_result["correct"]
            overall_total += dataset_result["total"]

        # Calculate overall average
        results["overall_accuracy"] = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
        results["overall_correct"] = overall_correct
        results["overall_total"] = overall_total

        return results


class VLLMEvaluator(BaseMedicalEvaluator):
    """Evaluates models on medical datasets using local vLLM."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")

        self.model_path = model_path
        model_name = os.path.basename(model_path)
        super().__init__(model_name)

        print(f"\n🚀 Loading model with vLLM: {self.model_name}")
        print(f"   Path: {model_path}")

        # Initialize vLLM
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": True,
        }

        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len
        if quantization:
            llm_kwargs["quantization"] = quantization

        self.llm = LLM(**llm_kwargs)

        # Get tokenizer for chat template
        self.tokenizer = self.llm.get_tokenizer()

        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            stop=["\n", "###", "User:", "Assistant:"],
        )

        print(f"   ✓ Model loaded successfully\n")

    def generate_responses(
        self,
        prompts: List[List[Dict[str, str]]],
        batch_size: int = 32,
    ) -> List[str]:
        """Generate responses using vLLM."""
        # Convert chat messages to formatted prompts
        formatted_prompts = []
        for messages in prompts:
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for non-chat models
                prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            formatted_prompts.append(prompt)

        # Generate with vLLM
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


class OpenAIEvaluator(BaseMedicalEvaluator):
    """Evaluates models on medical datasets using OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not installed. Install with: pip install openai"
            )

        super().__init__(model_name)
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )

        print(f"\n🌐 Using OpenAI-compatible API")
        print(f"   API Base: {api_base}")
        print(f"   Model: {model_name}")

        # Test connection
        try:
            # Try to get model info (works with vLLM)
            models = self.client.models.list()
            available_models = [m.id for m in models.data]
            if model_name not in available_models and available_models:
                print(f"   ⚠ Model '{model_name}' not in available models: {available_models}")
            print(f"   ✓ API connection successful\n")
        except Exception as e:
            print(f"   ⚠ API test failed: {e}")
            print(f"   Will attempt to proceed...\n")

    def generate_responses(
        self,
        prompts: List[List[Dict[str, str]]],
        batch_size: int = 32,
    ) -> List[str]:
        """Generate responses using OpenAI-compatible API."""
        responses = []

        for messages in prompts:
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    responses.append(response.choices[0].message.content)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"   Retry {attempt + 1}/{self.max_retries} after error: {e}")
                        time.sleep(self.retry_delay * (attempt + 1))
                    else:
                        print(f"   Failed after {self.max_retries} attempts: {e}")
                        responses.append("")  # Empty response on failure

        return responses


def format_results_table(results: Dict[str, Any]) -> str:
    """Format results as a markdown table."""
    lines = [
        "\n" + "=" * 60,
        f"📋 RESULTS FOR: {results['model']}",
        "=" * 60,
        "",
        "| Dataset | Correct | Total | Accuracy |",
        "|---------|---------|-------|----------|",
    ]

    for name, data in results["datasets"].items():
        lines.append(
            f"| {data['dataset']:<30} | {data['correct']:>7} | {data['total']:>5} | {data['accuracy']:>7.2f}% |"
        )

    lines.append("|---------|---------|-------|----------|")
    lines.append(
        f"| {'OVERALL':<30} | {results['overall_correct']:>7} | {results['overall_total']:>5} | {results['overall_accuracy']:>7.2f}% |"
    )
    lines.append("\n")

    return "\n".join(lines)


def create_evaluator(args) -> BaseMedicalEvaluator:
    """Create appropriate evaluator based on arguments."""

    # Check if using API mode
    if args.api_base:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package required for API mode. Install with: pip install openai"
            )

        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key and "localhost" not in args.api_base and "127.0.0.1" not in args.api_base:
            print("⚠ Warning: No API key provided. Set --api_key or OPENAI_API_KEY environment variable.")

        return OpenAIEvaluator(
            api_base=args.api_base,
            model_name=args.model_name or args.model_path,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

    # Local vLLM mode
    elif args.model_path:
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM package required for local mode. Install with: pip install vllm"
            )

        return VLLMEvaluator(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            quantization=args.quantization,
        )

    else:
        raise ValueError("Either --model_path (for local vLLM) or --api_base (for API mode) is required")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on OpenMedicalLLM Leaderboard datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local vLLM mode
  python open_medical_llm_eval.py --model_path /path/to/model --output_dir results/

  # vLLM server / OpenAI-compatible API mode
  python open_medical_llm_eval.py --api_base http://localhost:8000/v1 --model_name llama-3-8b

  # OpenAI API
  python open_medical_llm_eval.py --api_base https://api.openai.com/v1 --model_name gpt-4 --api_key $OPENAI_API_KEY

  # With specific datasets
  python open_medical_llm_eval.py --api_base http://localhost:8000/v1 --model_name llama-3-8b \\
      --datasets medqa pubmedqa --batch_size 16
        """
    )

    # Model source arguments (mutually exclusive modes)
    model_group = parser.add_argument_group("Model Source (choose one)")
    model_group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to local model (for local vLLM mode)"
    )
    model_group.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (e.g., http://localhost:8000/v1)"
    )

    # API-specific arguments
    api_group = parser.add_argument_group("API Mode Options (required with --api_base)")
    api_group.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for API calls (e.g., 'gpt-4', 'llama-3-8b')"
    )
    api_group.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (or set OPENAI_API_KEY environment variable)"
    )
    api_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for API calls"
    )
    api_group.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate per request"
    )
    api_group.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for failed API calls"
    )
    api_group.add_argument(
        "--retry_delay",
        type=float,
        default=1.0,
        help="Delay between retries (seconds)"
    )

    # Local vLLM-specific arguments
    local_group = parser.add_argument_group("Local vLLM Options (used with --model_path)")
    local_group.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    local_group.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0 to 1.0)"
    )
    local_group.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model context length"
    )
    local_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "bfloat16", "float"],
        help="Data type for model weights"
    )
    local_group.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm"],
        help="Quantization method used for the model"
    )

    # Common arguments
    common_group = parser.add_argument_group("Common Options")
    common_group.add_argument(
        "--datasets",
        nargs="+",
        default=["medqa", "pubmedqa", "medmcqa", "mmlu_anatomy", "mmlu_professional_medicine"],
        help="List of datasets to evaluate on"
    )
    common_group.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    common_group.add_argument(
        "--cache_dir",
        type=str,
        default="./datasets_cache",
        help="Directory to cache downloaded datasets"
    )
    common_group.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    common_group.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save detailed predictions to JSON"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model_path and not args.api_base:
        parser.error("Either --model_path or --api_base is required")

    if args.api_base and not args.model_name:
        parser.error("--model_name is required when using --api_base")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize dataset loader
    loader = MedicalDatasetLoader(cache_dir=args.cache_dir)

    # Create appropriate evaluator
    evaluator = create_evaluator(args)

    # Run evaluation
    results = evaluator.evaluate_all(
        dataset_names=args.datasets,
        loader=loader,
        batch_size=args.batch_size,
    )

    # Print results table
    table = format_results_table(results)
    print(table)

    # Save results
    model_safe_name = re.sub(r'[^\w\-]', '_', results['model'])
    output_file = os.path.join(args.output_dir, f"medical_eval_{model_safe_name}.json")

    # Optionally remove detailed predictions to save space
    if not args.save_predictions:
        for ds_result in results["datasets"].values():
            ds_result.pop("predictions", None)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: {output_file}")

    # Also save a summary CSV
    csv_file = os.path.join(args.output_dir, f"medical_eval_{model_safe_name}.csv")
    with open(csv_file, "w") as f:
        f.write("Dataset,Correct,Total,Accuracy\n")
        for name, data in results["datasets"].items():
            f.write(f"{data['dataset']},{data['correct']},{data['total']},{data['accuracy']:.2f}\n")
        f.write(f"OVERALL,{results['overall_correct']},{results['overall_total']},{results['overall_accuracy']:.2f}\n")

    print(f"💾 CSV summary saved to: {csv_file}")


if __name__ == "__main__":
    main()
