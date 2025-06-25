#!/usr/bin/env python3
"""
Test script specifically for Llama 3 8B performance
Compares GPT-2 vs Llama 3 performance on reasoning tasks
"""

import sys
import os
import time
import json
import torch
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from reasoning.runner import ToTRunner
from model_config import ModelConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_system_readiness():
    """Check if system is ready for Llama 3"""
    print("SYSTEM READINESS CHECK")
    print("="*50)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM: {vram_gb:.1f}GB")
        
        if vram_gb < 12:
            print("⚠ WARNING: Low VRAM. Consider using quantized models.")
        elif vram_gb >= 16:
            print("✓ Sufficient VRAM for full Llama 3 8B")
        else:
            print("⚠ Marginal VRAM. Monitor memory usage.")
    else:
        print("⚠ No CUDA. Llama 3 will be very slow on CPU.")
    
    # Check HuggingFace authentication
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✓ HuggingFace authenticated as: {user['name']}")
        auth_ok = True
    except Exception as e:
        print(f"✗ HuggingFace authentication failed: {e}")
        print("Run: huggingface-cli login")
        auth_ok = False
    
    return cuda_available, auth_ok


def test_model_comparison():
    """Compare GPT-2 vs Llama 3 on the same task"""
    print("\n" + "="*80)
    print("MODEL COMPARISON: GPT-2 vs LLAMA 3")
    print("="*80)
    
    task_id = "geometry_001"
    results = {}
    
    # Test GPT-2 (baseline)
    print("\n1. Testing GPT-2 (baseline):")
    try:
        gpt2_runner = ToTRunner(
            model_name="gpt2",
            num_paths=3,
            max_depth=2
        )
        
        start_time = time.time()
        gpt2_result = gpt2_runner.run_single_task(task_id, save_results=False)
        gpt2_time = time.time() - start_time
        
        results["gpt2"] = {
            "result": gpt2_result,
            "time": gpt2_time,
            "model": "gpt2"
        }
        
        print(f"   Answer: {gpt2_result.get('final_answer', 'No answer')[:100]}...")
        print(f"   Consensus: {gpt2_result.get('consensus_score', 0):.3f}")
        print(f"   Time: {gpt2_time:.2f}s")
        
    except Exception as e:
        print(f"   Error: {e}")
        results["gpt2"] = {"error": str(e)}
    
    # Test Llama 3 8B
    print("\n2. Testing Llama 3 8B:")
    try:
        llama_runner = ToTRunner(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            num_paths=5,
            max_depth=3
        )
        
        start_time = time.time()
        llama_result = llama_runner.run_single_task(task_id, save_results=False)
        llama_time = time.time() - start_time
        
        results["llama3"] = {
            "result": llama_result,
            "time": llama_time,
            "model": "meta-llama/Meta-Llama-3-8B-Instruct"
        }
        
        print(f"   Answer: {llama_result.get('final_answer', 'No answer')[:100]}...")
        print(f"   Consensus: {llama_result.get('consensus_score', 0):.3f}")
        print(f"   Time: {llama_time:.2f}s")
        
    except Exception as e:
        print(f"   Error: {e}")
        results["llama3"] = {"error": str(e)}
    
    # Compare results
    if "gpt2" in results and "llama3" in results and "error" not in results["gpt2"] and "error" not in results["llama3"]:
        print("\n3. COMPARISON:")
        
        gpt2_consensus = results["gpt2"]["result"].get("consensus_score", 0)
        llama_consensus = results["llama3"]["result"].get("consensus_score", 0)
        
        consensus_improvement = ((llama_consensus - gpt2_consensus) / gpt2_consensus * 100) if gpt2_consensus > 0 else 0
        time_ratio = results["llama3"]["time"] / results["gpt2"]["time"] if results["gpt2"]["time"] > 0 else 0
        
        print(f"   Consensus Score: {gpt2_consensus:.3f} → {llama_consensus:.3f} ({consensus_improvement:+.1f}%)")
        print(f"   Time Ratio: {time_ratio:.1f}x (Llama vs GPT-2)")
        
        # Quality assessment
        gpt2_answer = results["gpt2"]["result"].get("final_answer", "").lower()
        llama_answer = results["llama3"]["result"].get("final_answer", "").lower()
        
        print(f"\n   Quality Assessment:")
        print(f"   GPT-2 Answer Quality: {'Poor' if len(gpt2_answer) < 10 or 'area' not in gpt2_answer else 'Reasonable'}")
        print(f"   Llama Answer Quality: {'Poor' if len(llama_answer) < 10 or 'area' not in llama_answer else 'Reasonable'}")
    
    return results


def test_llama_domains():
    """Test Llama 3 across different domains"""
    print("\n" + "="*80)
    print("LLAMA 3 DOMAIN PERFORMANCE TEST")
    print("="*80)
    
    domains = ["geometry", "math", "logic", "probability"]
    results = {}
    
    runner = ToTRunner(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        num_paths=5,
        max_depth=3
    )
    
    for domain in domains:
        print(f"\nTesting {domain.upper()} domain:")
        
        try:
            domain_tasks = runner.task_loader.get_tasks_by_domain(domain)
            if not domain_tasks:
                print(f"   No tasks found for {domain}")
                continue
            
            task_id = domain_tasks[0]["id"]
            print(f"   Task: {task_id}")
            
            start_time = time.time()
            result = runner.run_single_task(task_id, save_results=False)
            execution_time = time.time() - start_time
            
            results[domain] = {
                "task_id": task_id,
                "result": result,
                "execution_time": execution_time
            }
            
            answer = result.get('final_answer', 'No answer')
            print(f"   Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
            print(f"   Consensus: {result.get('consensus_score', 0):.3f}")
            print(f"   Time: {execution_time:.2f}s")
            
            # Domain-specific quality check
            quality_score = assess_domain_quality(answer, domain)
            print(f"   Quality: {quality_score}/5")
            
        except Exception as e:
            print(f"   Error: {e}")
            results[domain] = {"error": str(e)}
    
    return results


def assess_domain_quality(answer: str, domain: str) -> int:
    """Assess answer quality for specific domain (1-5 scale)"""
    answer_lower = answer.lower()
    
    if domain == "geometry":
        score = 1
        if "area" in answer_lower: score += 1
        if "hypotenuse" in answer_lower: score += 1
        if any(unit in answer_lower for unit in ["cm", "cm²", "square"]): score += 1
        if any(num in answer_lower for num in ["30", "13"]): score += 1
        return min(5, score)
    
    elif domain == "math":
        score = 1
        if any(char.isdigit() for char in answer): score += 1
        if any(word in answer_lower for word in ["calculate", "equals", "result"]): score += 1
        if len(answer) > 20: score += 1
        if "=" in answer: score += 1
        return min(5, score)
    
    elif domain == "logic":
        score = 1
        if any(word in answer_lower for word in ["therefore", "because", "if", "then"]): score += 2
        if len(answer) > 30: score += 1
        if any(word in answer_lower for word in ["true", "false", "conclusion"]): score += 1
        return min(5, score)
    
    elif domain == "probability":
        score = 1
        if any(word in answer_lower for word in ["probability", "chance", "likely"]): score += 2
        if any(char in answer for char in [".", "/", "%"]): score += 1
        if len(answer) > 15: score += 1
        return min(5, score)
    
    return 3  # Default score


def run_llama_test():
    """Run Llama 3 performance test"""
    print("LLAMA 3 8B PERFORMANCE TEST")
    print("="*50)

    # Check system readiness
    cuda_ok, auth_ok = check_system_readiness()

    if not auth_ok:
        print("\n❌ Cannot proceed without HuggingFace authentication.")
        print("Please run: huggingface-cli login")
        return

    # Run comparison test
    results = test_model_comparison()

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    if "llama3" in results and "error" not in results["llama3"]:
        print("✓ Llama 3 8B working successfully")
        llama_consensus = results["llama3"]["result"].get("consensus_score", 0)
        print(f"✓ Consensus score: {llama_consensus:.3f}")
    else:
        print("❌ Llama 3 8B failed to load or run")
        if "llama3" in results:
            print(f"Error: {results['llama3'].get('error', 'Unknown error')}")

    return results


if __name__ == "__main__":
    run_llama_test()
