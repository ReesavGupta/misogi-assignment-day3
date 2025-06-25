"""
Model Configuration for Enhanced Performance
Provides configuration options for different models including quantized versions
"""

import torch
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for different model options"""
    
    # Model configurations with performance characteristics
    MODELS = {
        "gpt2": {
            "name": "gpt2",
            "type": "causal_lm",
            "size": "small",
            "vram_gb": 1,
            "performance": "basic",
            "reasoning_capability": "limited"
        },
        "gpt2-medium": {
            "name": "gpt2-medium", 
            "type": "causal_lm",
            "size": "medium",
            "vram_gb": 2,
            "performance": "improved",
            "reasoning_capability": "moderate"
        },
        "gpt2-large": {
            "name": "gpt2-large",
            "type": "causal_lm", 
            "size": "large",
            "vram_gb": 3,
            "performance": "good",
            "reasoning_capability": "better"
        },
        "distilgpt2": {
            "name": "distilgpt2",
            "type": "causal_lm",
            "size": "small",
            "vram_gb": 0.5,
            "performance": "fast",
            "reasoning_capability": "limited"
        },
        # Llama 3 models
        "llama3-8b-instruct": {
            "name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "type": "causal_lm",
            "size": "large",
            "vram_gb": 16,
            "performance": "excellent",
            "reasoning_capability": "high",
            "requires_auth": True,
            "note": "Requires HuggingFace authentication and significant VRAM"
        },
        "llama3-8b": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "type": "causal_lm",
            "size": "large",
            "vram_gb": 16,
            "performance": "excellent",
            "reasoning_capability": "high",
            "requires_auth": True,
            "note": "Base model, requires HuggingFace authentication"
        },
        # Alternative high-performance models (no auth required)
        "mistral-7b-instruct": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "type": "causal_lm",
            "size": "large",
            "vram_gb": 4,
            "performance": "excellent",
            "reasoning_capability": "high",
            "requires_auth": False,
            "note": "Excellent reasoning, fits in 4GB VRAM with quantization"
        },
        "phi3-mini": {
            "name": "microsoft/Phi-3-mini-4k-instruct",
            "type": "causal_lm",
            "size": "small",
            "vram_gb": 2,
            "performance": "very_good",
            "reasoning_capability": "high",
            "requires_auth": False,
            "note": "Compact but powerful, great for limited VRAM"
        },
        "llama3-8b-q4": {
            "name": "microsoft/Llama-3-8B-Instruct-GGUF",
            "type": "gguf",
            "size": "medium",
            "vram_gb": 6,
            "performance": "very_good",
            "reasoning_capability": "high",
            "requires_auth": False,
            "note": "Quantized version, lower VRAM requirements"
        }
    }
    
    @classmethod
    def get_recommended_model(cls, available_vram_gb: float = 4.0, prefer_llama: bool = True) -> str:
        """
        Get recommended model based on available VRAM

        Args:
            available_vram_gb: Available VRAM in GB
            prefer_llama: Whether to prefer Llama models over GPT-2

        Returns:
            Recommended model name
        """
        suitable_models = []

        for model_key, config in cls.MODELS.items():
            if config["vram_gb"] <= available_vram_gb:
                suitable_models.append((model_key, config))

        if not suitable_models:
            logger.warning(f"No suitable models found for {available_vram_gb}GB VRAM, using distilgpt2")
            return "distilgpt2"

        # Sort by reasoning capability and performance
        capability_order = {"limited": 1, "moderate": 2, "better": 3, "high": 4}

        # If prefer_llama is True, prioritize Llama models
        if prefer_llama:
            llama_models = [(k, v) for k, v in suitable_models if "llama" in k]
            if llama_models:
                llama_models.sort(key=lambda x: capability_order.get(x[1]["reasoning_capability"], 0), reverse=True)
                recommended = llama_models[0][0]
                logger.info(f"Recommended Llama model for {available_vram_gb}GB VRAM: {recommended}")
                return recommended

        # Fallback to best available model
        suitable_models.sort(key=lambda x: capability_order.get(x[1]["reasoning_capability"], 0), reverse=True)
        recommended = suitable_models[0][0]
        logger.info(f"Recommended model for {available_vram_gb}GB VRAM: {recommended}")

        return recommended
    
    @classmethod
    def get_model_info(cls, model_key: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        return cls.MODELS.get(model_key, cls.MODELS["gpt2"])
    
    @classmethod
    def get_optimized_generation_params(cls, model_key: str, task_domain: str = "general") -> Dict[str, Any]:
        """
        Get optimized generation parameters for a specific model and domain
        
        Args:
            model_key: Model identifier
            task_domain: Task domain (geometry, math, logic, etc.)
            
        Returns:
            Dictionary of generation parameters
        """
        base_params = {
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
        
        # Model-specific adjustments
        model_adjustments = {
            "gpt2": {
                "temperature": 0.8,
                "top_p": 0.9,
                "max_length": 150
            },
            "gpt2-medium": {
                "temperature": 0.75,
                "top_p": 0.87,
                "max_length": 200
            },
            "gpt2-large": {
                "temperature": 0.7,
                "top_p": 0.85,
                "max_length": 250
            },
            "distilgpt2": {
                "temperature": 0.85,
                "top_p": 0.92,
                "max_length": 120
            },
            # Llama 3 models
            "llama3-8b-instruct": {
                "temperature": 0.6,
                "top_p": 0.8,
                "top_k": 40,
                "max_length": 400,
                "repetition_penalty": 1.05
            },
            "llama3-8b": {
                "temperature": 0.65,
                "top_p": 0.82,
                "top_k": 40,
                "max_length": 400,
                "repetition_penalty": 1.05
            },
            "llama3-8b-q4": {
                "temperature": 0.65,
                "top_p": 0.82,
                "top_k": 40,
                "max_length": 350,
                "repetition_penalty": 1.05
            }
        }
        
        # Domain-specific adjustments
        domain_adjustments = {
            "math": {
                "temperature": 0.6,  # More deterministic for math
                "top_p": 0.8
            },
            "geometry": {
                "temperature": 0.65,
                "top_p": 0.82
            },
            "logic": {
                "temperature": 0.6,
                "top_p": 0.8
            },
            "probability": {
                "temperature": 0.7,
                "top_p": 0.85
            },
            "code": {
                "temperature": 0.5,  # Very deterministic for code
                "top_p": 0.75
            }
        }
        
        # Combine parameters
        params = base_params.copy()
        
        if model_key in model_adjustments:
            params.update(model_adjustments[model_key])
            
        if task_domain in domain_adjustments:
            params.update(domain_adjustments[task_domain])
            
        return params
    
    @classmethod
    def check_system_compatibility(cls) -> Dict[str, Any]:
        """
        Check system compatibility and recommend settings
        
        Returns:
            System compatibility information
        """
        compatibility = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "recommended_device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        if torch.cuda.is_available():
            try:
                # Get VRAM info for first GPU
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                compatibility["vram_gb"] = gpu_memory / (1024**3)
                compatibility["recommended_model"] = cls.get_recommended_model(compatibility["vram_gb"])
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
                compatibility["vram_gb"] = 4.0  # Conservative estimate
                compatibility["recommended_model"] = "gpt2"
        else:
            compatibility["vram_gb"] = 0
            compatibility["recommended_model"] = "distilgpt2"  # Smallest model for CPU
            
        return compatibility
    
    @classmethod
    def print_system_info(cls):
        """Print system compatibility information"""
        info = cls.check_system_compatibility()
        
        print("\n" + "="*60)
        print("SYSTEM COMPATIBILITY CHECK")
        print("="*60)
        print(f"CUDA Available: {info['cuda_available']}")
        print(f"GPU Count: {info['device_count']}")
        print(f"Recommended Device: {info['recommended_device']}")
        print(f"Available VRAM: {info['vram_gb']:.1f} GB")
        print(f"Recommended Model: {info['recommended_model']}")
        
        model_info = cls.get_model_info(info['recommended_model'])
        print(f"Model Performance: {model_info['performance']}")
        print(f"Reasoning Capability: {model_info['reasoning_capability']}")
        print("="*60)


if __name__ == "__main__":
    # Test the configuration
    ModelConfig.print_system_info()
    
    # Show generation parameters for different domains
    print("\nOptimized Generation Parameters:")
    for domain in ["math", "geometry", "logic", "code"]:
        params = ModelConfig.get_optimized_generation_params("gpt2", domain)
        print(f"{domain}: temp={params['temperature']}, top_p={params['top_p']}")
