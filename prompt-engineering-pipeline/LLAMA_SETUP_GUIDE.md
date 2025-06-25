# Llama 3 8B Setup Guide

## Overview

This guide will help you set up Llama 3 8B with GPU acceleration for dramatically improved reasoning performance in your prompt engineering pipeline.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 20GB+ free space for model download
- **CUDA**: CUDA 11.8+ or 12.x

### Minimum Requirements (with quantization)
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **RAM**: 16GB+ system RAM
- **CUDA**: CUDA 11.8+

## Step 1: Install Dependencies

```bash
# Install/upgrade PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install -r requirements.txt

# Or install individually:
pip install transformers>=4.36.0
pip install accelerate>=0.20.0
pip install bitsandbytes>=0.41.0
pip install sentencepiece>=0.1.99
pip install huggingface_hub>=0.16.0
```

## Step 2: HuggingFace Authentication

### Option A: CLI Login (Recommended)
```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Login with your token
huggingface-cli login
```

### Option B: Environment Variable
```bash
# Set your HuggingFace token
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### Getting Access to Llama 3
1. Create account at https://huggingface.co/
2. Request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. Create access token at https://huggingface.co/settings/tokens
4. Use token for authentication

## Step 3: Verify Setup

```bash
# Run setup verification
python setup_llama.py

# Test system compatibility
python -c "from src.model_config import ModelConfig; ModelConfig.print_system_info()"

# Quick Llama test
python test_llama.py
```

## Step 4: Usage Examples

### Basic Usage
```bash
# Run with Llama 3 8B (default)
python src/reasoning/runner.py --task geometry_001

# Specify model explicitly
python src/reasoning/runner.py --model meta-llama/Meta-Llama-3-8B-Instruct --task math_001
```

### High-Quality Reasoning
```bash
# Maximum quality (slower)
python src/reasoning/runner.py --paths 6 --depth 4 --task logic_001

# Balanced performance
python src/reasoning/runner.py --paths 5 --depth 3 --task probability_001
```

### Domain-Specific Testing
```bash
# Test all geometry problems
python src/reasoning/runner.py --domain geometry

# Test all domains
python src/reasoning/runner.py --all
```

## Performance Optimization

### GPU Memory Management
```python
# Monitor GPU usage
nvidia-smi

# For limited VRAM, use quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)
```

### Generation Parameters
The pipeline automatically optimizes parameters for Llama 3:
- **Temperature**: 0.6 (more focused than GPT-2)
- **Top-p**: 0.8 (nucleus sampling)
- **Top-k**: 40 (token selection)
- **Max tokens**: 400 (longer responses)

## Expected Performance Improvements

### Compared to GPT-2
- **Reasoning Quality**: 3-5x improvement
- **Answer Accuracy**: 40-70% vs 0-20%
- **Coherence**: 75-85% vs 45%
- **Domain Understanding**: Significantly better
- **Hallucination Rate**: 15-25% vs 50%

### Benchmark Results
| Domain | GPT-2 Accuracy | Llama 3 Expected |
|--------|----------------|------------------|
| Geometry | 0% | 60-80% |
| Math | 15% | 70-85% |
| Logic | 10% | 65-80% |
| Probability | 20% | 60-75% |
| Code | 7% | 75-90% |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce reasoning paths
python src/reasoning/runner.py --paths 3 --depth 2

# Use quantization (modify tot_engine.py)
load_in_4bit=True
```

### Authentication Issues
```bash
# Check authentication
huggingface-cli whoami

# Re-login if needed
huggingface-cli login --token your_token_here
```

### Slow Performance
```bash
# Verify CUDA is being used
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU utilization
nvidia-smi -l 1
```

### Model Loading Errors
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Re-download model
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')"
```

## Alternative Models

### If 16GB VRAM is not available:
1. **Quantized Llama 3**: Use 4-bit quantization (6-8GB VRAM)
2. **Llama 3.1 8B**: Similar performance, potentially better efficiency
3. **Code Llama**: Specialized for code-related tasks
4. **Mistral 7B**: Good alternative with lower VRAM requirements

### Model Configuration Examples:
```python
# Full precision (16GB VRAM)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Quantized (6-8GB VRAM)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

# Alternative model (lower VRAM)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
```

## Monitoring and Optimization

### Performance Monitoring
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Pipeline-specific monitoring
python test_improvements.py --test all
```

### Optimization Tips
1. **Batch Processing**: Process multiple tasks together
2. **Caching**: Enable model caching for repeated runs
3. **Mixed Precision**: Use float16 for faster inference
4. **Gradient Checkpointing**: Reduce memory usage
5. **Dynamic Batching**: Adjust batch size based on available memory

## Integration with Existing Pipeline

The Llama 3 integration is designed to be drop-in compatible:

1. **Automatic Detection**: Pipeline detects Llama models and adjusts parameters
2. **Domain Optimization**: Uses domain-specific prompts and generation settings
3. **Verification Layer**: Enhanced answer verification for better quality
4. **Backward Compatibility**: Still supports GPT-2 models as fallback

## Next Steps

1. **Run Setup**: Execute `python setup_llama.py`
2. **Test Performance**: Run `python test_llama.py`
3. **Compare Results**: Use `python test_improvements.py`
4. **Optimize Settings**: Adjust paths/depth based on your hardware
5. **Monitor Performance**: Track improvements in evaluation reports

## Support

For issues:
1. Check CUDA compatibility: `nvidia-smi`
2. Verify authentication: `huggingface-cli whoami`
3. Test model loading: `python test_llama.py`
4. Review logs in `logs/` directory
5. Adjust parameters based on available VRAM
