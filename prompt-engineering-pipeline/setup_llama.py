#!/usr/bin/env python3
"""
Setup script for Llama 3 8B integration
Installs required dependencies and configures the environment
"""

import subprocess
import sys
import os
import torch
from pathlib import Path

def check_cuda():
    """Check CUDA availability and version"""
    print("Checking CUDA availability...")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            vram_gb = gpu_props.total_memory / (1024**3)
            print(f"GPU {i}: {gpu_props.name} ({vram_gb:.1f}GB VRAM)")
            
        return True, vram_gb
    else:
        print("CUDA not available. Llama 3 8B requires GPU for optimal performance.")
        return False, 0

def install_dependencies():
    """Install required packages for Llama 3"""
    print("\nInstalling required dependencies...")
    
    packages = [
        "torch>=2.0.0",
        "transformers>=4.36.0", 
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",  # For quantization
        "sentencepiece>=0.1.99",  # For Llama tokenizer
        "protobuf>=3.20.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    return True

def setup_huggingface_auth():
    """Guide user through HuggingFace authentication setup"""
    print("\n" + "="*60)
    print("HUGGINGFACE AUTHENTICATION SETUP")
    print("="*60)
    
    print("\nTo use Llama 3 models, you need to:")
    print("1. Create a HuggingFace account at https://huggingface.co/")
    print("2. Request access to Llama 3 models at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
    print("3. Create an access token at https://huggingface.co/settings/tokens")
    print("4. Install HuggingFace CLI: pip install huggingface_hub")
    print("5. Login with: huggingface-cli login")
    
    print("\nAlternatively, you can use environment variable:")
    print("export HUGGINGFACE_HUB_TOKEN='your_token_here'")
    
    # Check if already authenticated
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"\n✓ Already authenticated as: {user['name']}")
        return True
    except Exception:
        print("\n⚠ Not authenticated yet. Please follow the steps above.")
        return False

def test_model_loading():
    """Test if Llama 3 model can be loaded"""
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"Testing model loading: {model_name}")
        
        # Test tokenizer first (faster)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded successfully")
        
        # Test model loading (this will be slow and memory intensive)
        print("Loading model (this may take several minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Model loaded successfully")
        
        # Test generation
        print("Testing generation...")
        test_prompt = "What is 2 + 2?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test response: {response}")
        print("✓ Model generation working")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("\nPossible solutions:")
        print("1. Ensure you have HuggingFace authentication set up")
        print("2. Check if you have sufficient VRAM (16GB+ recommended)")
        print("3. Try using a quantized model instead")
        return False

def create_llama_config():
    """Create configuration file for Llama 3 setup"""
    config_content = """# Llama 3 Configuration

## Model Options
- **Full Model**: meta-llama/Meta-Llama-3-8B-Instruct (16GB VRAM)
- **Quantized**: Use 4-bit quantization for 6-8GB VRAM
- **Alternative**: Use smaller models if VRAM is limited

## Usage Examples

### Basic Usage
```bash
python src/reasoning/runner.py --task geometry_001
```

### With specific model
```bash
python src/reasoning/runner.py --model meta-llama/Meta-Llama-3-8B-Instruct --task math_001
```

### High-quality reasoning
```bash
python src/reasoning/runner.py --paths 6 --depth 4 --task logic_001
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: --paths 3
- Use quantization: Add load_in_4bit=True to model loading
- Use gradient checkpointing

### Authentication Issues
- Run: huggingface-cli login
- Or set: export HUGGINGFACE_HUB_TOKEN='your_token'

### Slow Performance
- Ensure CUDA is available
- Use torch.compile() for faster inference
- Consider using vLLM for production
"""
    
    config_path = Path("LLAMA_SETUP.md")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"\n✓ Configuration guide saved to {config_path}")

def main():
    """Main setup function"""
    print("LLAMA 3 8B SETUP FOR PROMPT ENGINEERING PIPELINE")
    print("="*60)
    
    # Check CUDA
    cuda_available, vram_gb = check_cuda()
    
    if not cuda_available:
        print("\n⚠ WARNING: CUDA not available. Llama 3 8B will be very slow on CPU.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    elif vram_gb < 12:
        print(f"\n⚠ WARNING: Only {vram_gb:.1f}GB VRAM available.")
        print("Llama 3 8B typically requires 16GB+ VRAM.")
        print("Consider using quantized models or smaller variants.")
        response = input("Continue with quantization? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please check your environment.")
        return
    
    # Setup authentication
    setup_huggingface_auth()
    
    # Create configuration
    create_llama_config()
    
    # Test model loading (optional)
    print("\n" + "="*60)
    response = input("Test model loading now? This will download ~16GB (y/N): ")
    if response.lower() == 'y':
        test_model_loading()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Ensure HuggingFace authentication is set up")
    print("2. Test with: python src/reasoning/runner.py --task geometry_001")
    print("3. Check LLAMA_SETUP.md for detailed usage instructions")
    print("4. Monitor GPU memory usage during execution")

if __name__ == "__main__":
    main()
