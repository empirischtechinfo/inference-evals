#!/bin/bash

# =============================================================================
# Fine-tuning Environment Setup Script
# =============================================================================
# This script sets up a complete conda environment for LLM fine-tuning with:
# - System dependencies (build tools, CUDA prerequisites)
# - Miniconda installation
# - Conda environment creation with Python 3.10
# - PyTorch with CUDA 13.0 support
# - Fine-tuning libraries (transformers, peft, trl, accelerate, etc.)
# - GPTQ quantization support
# - Evaluation tools (lm-eval)
#
# Usage:
#   chmod +x setup_finetuning_env.sh
#   ./setup_finetuning_env.sh
#
# Or run non-interactively:
#   ./setup_finetuning_env.sh --yes
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV_NAME="finetuning"
PYTHON_VERSION="3.12"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}"
SKIP_SYSTEM_UPDATE=false
AUTO_YES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-system-update)
            SKIP_SYSTEM_UPDATE=true
            shift
            ;;
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        --env-name)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --yes                   Auto-answer yes to all prompts"
            echo "  --skip-system-update    Skip apt update/upgrade"
            echo "  --env-name NAME         Set conda environment name (default: finetuning)"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
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

confirm() {
    if [ "$AUTO_YES" = true ]; then
        return 0
    fi
    read -p "$1 [Y/n]: " response
    response=${response:-Y}
    if [[ "$response" =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# 1. System Dependencies
# =============================================================================

install_system_deps() {
    print_header "Step 1: Installing System Dependencies"

    if [ "$SKIP_SYSTEM_UPDATE" = false ]; then
        if confirm "Update and upgrade system packages?"; then
            echo "Updating package lists..."
            sudo apt update

            echo "Upgrading packages..."
            sudo apt upgrade -y
        fi
    else
        print_warning "Skipping system update as requested"
    fi

    echo "Installing build essentials..."
    sudo apt install -y build-essential cmake ninja-build

    echo "Installing GCC 12..."
    sudo apt install -y g++-12

    # Set LD_PRELOAD for libstdc++
    if ! grep -q "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ~/.bashrc; then
        echo "Setting up LD_PRELOAD in .bashrc..."
        echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' >> ~/.bashrc
        print_success "LD_PRELOAD configured in .bashrc"
    else
        print_warning "LD_PRELOAD already configured"
    fi

    # Apply to current session
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

    print_success "System dependencies installed"
}

# =============================================================================
# 2. Miniconda Installation
# =============================================================================

install_miniconda() {
    print_header "Step 2: Installing Miniconda"

    # Check if conda is already installed
    if command -v conda &> /dev/null; then
        print_warning "Conda is already installed at: $(which conda)"
        if confirm "Reinstall Miniconda?"; then
            echo "Proceeding with reinstallation..."
        else
            print_success "Using existing Miniconda installation"
            return 0
        fi
    fi

    # Download Miniconda
    if [ -f "$MINICONDA_INSTALLER" ]; then
        print_warning "Miniconda installer already downloaded"
        if confirm "Redownload?"; then
            rm -f "$MINICONDA_INSTALLER"
            echo "Downloading Miniconda..."
            wget "$MINICONDA_URL"
        fi
    else
        echo "Downloading Miniconda..."
        wget "$MINICONDA_URL"
    fi

    # Install Miniconda
    echo "Installing Miniconda..."
    if [ "$AUTO_YES" = true ]; then
        bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    else
        bash "$MINICONDA_INSTALLER"
    fi

    # Initialize conda for bash
    echo "Initializing conda for bash..."
    "$HOME/miniconda3/bin/conda" init bash

    print_success "Miniconda installed. Please restart your shell or run: source ~/.bashrc"
}

# =============================================================================
# 3. Conda Environment Setup
# =============================================================================

setup_conda_env() {
    print_header "Step 3: Setting Up Conda Environment"

    # Ensure conda is available
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        source "$HOME/miniconda3/bin/activate"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        source "$HOME/anaconda3/bin/activate"
    else
        print_error "Conda not found. Please ensure Miniconda is installed."
        exit 1
    fi

    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_warning "Conda environment '${CONDA_ENV_NAME}' already exists"
        if confirm "Remove and recreate environment?"; then
            echo "Removing existing environment..."
            conda deactivate || true
            conda env remove -n "$CONDA_ENV_NAME" -y
        else
            print_success "Using existing environment"
            source "$HOME/miniconda3/bin/activate" "$CONDA_ENV_NAME"
            return 0
        fi
    fi

    # Create new environment
    echo "Creating conda environment: $CONDA_ENV_NAME with Python $PYTHON_VERSION"
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y

    # Activate environment
    source "$HOME/miniconda3/bin/activate" "$CONDA_ENV_NAME"

    print_success "Conda environment '${CONDA_ENV_NAME}' created and activated"
}

# =============================================================================
# 4. Python Packages Installation
# =============================================================================

install_python_packages() {
    print_header "Step 4: Installing Python Packages"

    # Ensure we're in the conda environment
    if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
        source "$HOME/miniconda3/bin/activate" "$CONDA_ENV_NAME"
    fi

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing PyTorch with CUDA 13.0..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

    echo "Installing fine-tuning libraries..."
    pip install transformers accelerate bitsandbytes trl peft python-dotenv

    echo "Installing additional utilities..."
    pip install langdetect tqdm datasets numpy scipy scikit-learn

    echo "Installing GPTQModel (for quantization)..."
    pip install -v gptqmodel --no-build-isolation || {
        print_warning "GPTQModel installation failed. You may need to install it manually."
        print_warning "Try: pip install auto-gptq as an alternative"
    }

    echo "Installing evaluation tools..."
    pip install "lm-eval>=0.4.7"

    echo "Installing vLLM (for inference serving)..."
    pip install vllm || {
        print_warning "vLLM installation failed. This is optional for fine-tuning."
    }

    print_success "Python packages installed"
}

# =============================================================================
# 5. Install Additional Requirements
# =============================================================================

install_additional_requirements() {
    print_header "Step 5: Installing Additional Requirements"

    # Check for requirements.txt in common locations
    REQUIREMENTS_FILES=(
        "./requirements.txt"
        "../requirements.txt"
        "../../requirements.txt"
        "./setup/requirements.txt"
    )

    FOUND_REQUIREMENTS=false
    for req_file in "${REQUIREMENTS_FILES[@]}"; do
        if [ -f "$req_file" ]; then
            print_success "Found requirements.txt at: $req_file"
            if confirm "Install packages from $req_file?"; then
                pip install -r "$req_file"
                print_success "Additional requirements installed from $req_file"
            fi
            FOUND_REQUIREMENTS=true
            break
        fi
    done

    if [ "$FOUND_REQUIREMENTS" = false ]; then
        print_warning "No requirements.txt found in standard locations"
    fi
}

# =============================================================================
# 6. Post-Installation Setup
# =============================================================================

post_installation() {
    print_header "Step 6: Post-Installation Setup"

    # Create activation script for easy environment access
    ACTIVATE_SCRIPT="activate_finetuning.sh"
    cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Quick activation script for finetuning environment
source ~/miniconda3/bin/activate finetuning
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
echo "Fine-tuning environment activated!"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
exec bash
EOF
    chmod +x "$ACTIVATE_SCRIPT"
    print_success "Created activation script: $ACTIVATE_SCRIPT"

    # Create a verification script
    VERIFY_SCRIPT="verify_installation.py"
    cat > "$VERIFY_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""Verify fine-tuning environment installation."""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("Fine-tuning Environment Verification")
    print("=" * 60)
    print()

    # Check Python version
    print(f"Python version: {sys.version}")
    print()

    # Check PyTorch and CUDA
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print()
    except ImportError:
        print("✗ PyTorch not installed!")
        sys.exit(1)

    # Check core fine-tuning libraries
    print("Core Libraries:")
    all_ok = True
    all_ok &= check_import("transformers", "transformers")
    all_ok &= check_import("peft")
    all_ok &= check_import("trl")
    all_ok &= check_import("accelerate")
    all_ok &= check_import("bitsandbytes")
    print()

    # Check optional libraries
    print("Optional Libraries:")
    check_import("gptqmodel", "gptqmodel (quantization)")
    check_import("vllm", "vllm (inference)")
    check_import("lm_eval", "lm-eval (evaluation)")
    check_import("datasets", "datasets (HF datasets)")
    print()

    if all_ok:
        print("=" * 60)
        print("✓ All core libraries installed successfully!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("✗ Some libraries are missing. Check the output above.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
    chmod +x "$VERIFY_SCRIPT"
    print_success "Created verification script: $VERIFY_SCRIPT"

    # Run verification
    echo ""
    echo "Running verification..."
    python "$VERIFY_SCRIPT" || true
}

# =============================================================================
# 7. Print Summary
# =============================================================================

print_summary() {
    print_header "Setup Complete!"

    echo ""
    echo -e "${GREEN}Your fine-tuning environment is ready!${NC}"
    echo ""
    echo "To activate the environment in the future, run:"
    echo "  source ~/miniconda3/bin/activate $CONDA_ENV_NAME"
    echo "  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    echo ""
    echo "Or use the quick activation script:"
    echo "  ./activate_finetuning.sh"
    echo ""
    echo "To verify your installation:"
    echo "  python verify_installation.py"
    echo ""
    echo "Environment details:"
    echo "  - Conda env: $CONDA_ENV_NAME"
    echo "  - Python: $PYTHON_VERSION"
    echo "  - CUDA: 13.0"
    echo ""
    echo "Key packages installed:"
    echo "  - PyTorch (with CUDA 13.0)"
    echo "  - Transformers"
    echo "  - PEFT (Parameter Efficient Fine-Tuning)"
    echo "  - TRL (Transformer Reinforcement Learning)"
    echo "  - Accelerate"
    echo "  - BitsAndBytes (quantization)"
    echo "  - GPTQModel (quantization)"
    echo "  - lm-eval (evaluation)"
    echo "  - vLLM (inference serving)"
    echo ""
    print_warning "Note: Please restart your shell or run 'source ~/.bashrc' to apply all changes"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "============================================================================="
    echo "                    Fine-tuning Environment Setup"
    echo "============================================================================="
    echo ""
    echo "This script will:"
    echo "  1. Install system dependencies (build tools, GCC 12)"
    echo "  2. Install Miniconda (if not present)"
    echo "  3. Create conda environment: $CONDA_ENV_NAME"
    echo "  4. Install PyTorch with CUDA 13.0"
    echo "  5. Install fine-tuning libraries (transformers, peft, trl, etc.)"
    echo "  6. Install additional requirements from requirements.txt (if found)"
    echo ""

    if [ "$AUTO_YES" = false ]; then
        if ! confirm "Continue with setup?"; then
            echo "Setup cancelled."
            exit 0
        fi
    fi

    install_system_deps
    install_miniconda
    setup_conda_env
    install_python_packages
    install_additional_requirements
    post_installation
    print_summary
}

# Run main function
main
