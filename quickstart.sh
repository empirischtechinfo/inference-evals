#!/bin/bash

# =============================================================================
# Inference Evaluations - Quick Start Script
# =============================================================================
# This script provides a guided quick start for setting up the environment
# and running your first medical benchmark evaluation.
#
# Usage:
#   chmod +x quickstart.sh
#   ./quickstart.sh
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if conda is installed
check_conda() {
    if command -v conda &> /dev/null; then
        print_success "Conda is installed"
        return 0
    else
        return 1
    fi
}

# Check if finetuning environment exists
check_env() {
    if conda env list | grep -q "^finetuning "; then
        return 0
    else
        return 1
    fi
}

# Step 1: Setup Environment
setup_environment() {
    print_header "Step 1: Environment Setup"

    if check_env; then
        print_success "Fine-tuning environment already exists"
        read -p "Do you want to re-create it? (y/N): " recreate
        if [[ ! "$recreate" =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi

    if ! check_conda; then
        echo "Conda is not installed. Running setup script..."
        cd setup
        ./setup_finetuning_env.sh
        cd ..
    else
        echo "Conda is installed. Running environment setup..."
        cd setup
        ./setup_finetuning_env.sh --skip-system-update
        cd ..
    fi

    print_success "Environment setup complete"
}

# Step 2: Activate Environment
activate_environment() {
    print_header "Step 2: Activating Environment"

    if ! check_env; then
        print_error "Fine-tuning environment not found. Please run setup first."
        exit 1
    fi

    echo "Activating finetuning environment..."
    source ~/miniconda3/bin/activate finetuning
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

    print_success "Environment activated"
    echo "Python: $(which python)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
}

# Step 3: Download Datasets
download_datasets() {
    print_header "Step 3: Downloading Medical Datasets"

    cd benchmarks/medical

    if [ -d "../../datasets_cache" ] && [ "$(ls -A ../../datasets_cache 2>/dev/null)" ]; then
        print_warning "Datasets may already be cached"
        read -p "Re-download datasets? (y/N): " redownload
        if [[ ! "$redownload" =~ ^[Yy]$ ]]; then
            cd ../..
            return 0
        fi
    fi

    echo "Downloading datasets (this may take a while)..."
    python download_datasets.py --cache_dir ../../datasets_cache

    cd ../..
    print_success "Datasets downloaded"
}

# Step 4: Run Evaluation
run_evaluation() {
    print_header "Step 4: Run Medical Benchmark Evaluation"

    cd benchmarks/medical

    echo ""
    echo "Choose evaluation mode:"
    echo "1) Local vLLM (requires model path)"
    echo "2) OpenAI-compatible API (requires API endpoint)"
    read -p "Enter choice (1 or 2): " mode

    echo ""
    echo "Available datasets:"
    echo "  - medqa (MedQA/USMLE)"
    echo "  - pubmedqa"
    echo "  - medmcqa"
    echo "  - mmlu_anatomy"
    echo "  - mmlu_professional_medicine"
    echo "  - mmlu_college_medicine"
    echo "  - mmlu_college_biology"
    echo ""
    read -p "Enter datasets to evaluate (space-separated, or 'all'): " datasets

    if [ "$datasets" = "all" ]; then
        datasets="medqa pubmedqa medmcqa mmlu_anatomy mmlu_professional_medicine"
    fi

    if [ "$mode" = "1" ]; then
        echo ""
        read -p "Enter model path (local path or HF model ID): " model_path

        echo ""
        echo "Running evaluation with local vLLM..."
        python open_medical_llm_eval.py \
            --model_path "$model_path" \
            --datasets $datasets \
            --output_dir ./results \
            --batch_size 16

    elif [ "$mode" = "2" ]; then
        echo ""
        read -p "Enter API base URL (e.g., http://localhost:8000/v1): " api_base
        read -p "Enter model name: " model_name
        read -p "Enter API key (or press Enter for none): " api_key

        echo ""
        echo "Running evaluation with API..."
        if [ -n "$api_key" ]; then
            python open_medical_llm_eval.py \
                --api_base "$api_base" \
                --model_name "$model_name" \
                --api_key "$api_key" \
                --datasets $datasets \
                --output_dir ./results \
                --batch_size 10
        else
            python open_medical_llm_eval.py \
                --api_base "$api_base" \
                --model_name "$model_name" \
                --datasets $datasets \
                --output_dir ./results \
                --batch_size 10
        fi
    else
        print_error "Invalid choice"
        cd ../..
        return 1
    fi

    cd ../..
    print_success "Evaluation complete! Results saved to benchmarks/medical/results/"
}

# Step 5: View Results
view_results() {
    print_header "Step 5: View Results"

    cd benchmarks/medical

    if [ -d "results" ] && [ "$(ls -A results/*.json 2>/dev/null)" ]; then
        echo ""
        echo "Available results:"
        ls -lh results/

        echo ""
        read -p "Compare multiple models? (y/N): " compare
        if [[ "$compare" =~ ^[Yy]$ ]]; then
            python compare_models.py --results_dir ./results
        fi
    else
        print_warning "No results found in benchmarks/medical/results/"
    fi

    cd ../..
}

# Main menu
main_menu() {
    while true; do
        print_header "Inference Evaluations - Quick Start"

        echo ""
        echo "What would you like to do?"
        echo ""
        echo "1) Full Setup (environment + datasets + evaluation)"
        echo "2) Setup Environment Only"
        echo "3) Download Datasets Only"
        echo "4) Run Evaluation Only"
        echo "5) Compare Existing Results"
        echo "6) Exit"
        echo ""
        read -p "Enter your choice (1-6): " choice

        case $choice in
            1)
                setup_environment
                activate_environment
                download_datasets
                run_evaluation
                view_results
                ;;
            2)
                setup_environment
                activate_environment
                ;;
            3)
                activate_environment
                download_datasets
                ;;
            4)
                activate_environment
                run_evaluation
                ;;
            5)
                activate_environment
                view_results
                ;;
            6)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please try again."
                ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
    done
}

# Print welcome message
clear
print_header "Welcome to Inference Evaluations!"

echo ""
echo "This quick start script will help you:"
echo "  1. Set up a complete fine-tuning environment with conda"
echo "  2. Download medical benchmark datasets"
echo "  3. Run evaluations on your models"
echo "  4. Compare results across multiple models"
echo ""
echo "Requirements:"
echo "  - Linux system (WSL2 works)"
echo "  - NVIDIA GPU with CUDA support (for local evaluation)"
echo "  - Internet connection for downloading packages and datasets"
echo "  - Sudo access (for installing system dependencies)"
echo ""
read -p "Press Enter to continue or Ctrl+C to exit..."

# Run main menu
main_menu
