#!/usr/bin/env python3
"""
Test script to validate pipeline improvements
Runs before/after comparisons and performance analysis
"""

import sys
import os
import time
import json
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


def test_single_task_improvement():
    """Test improvements on a single task"""
    print("\n" + "="*80)
    print("TESTING SINGLE TASK IMPROVEMENT")
    print("="*80)
    
    # Test with original settings (baseline)
    print("\n1. Testing with BASELINE settings:")
    print("   - Paths: 3, Depth: 2, Model: gpt2")
    
    baseline_runner = ToTRunner(
        model_name="gpt2",
        num_paths=3,
        max_depth=2
    )
    
    start_time = time.time()
    baseline_result = baseline_runner.run_single_task("geometry_001", save_results=False)
    baseline_time = time.time() - start_time
    
    print(f"   Result: {baseline_result.get('final_answer', 'No answer')}")
    print(f"   Consensus: {baseline_result.get('consensus_score', 0):.3f}")
    print(f"   Time: {baseline_time:.2f}s")
    
    # Test with improved settings
    print("\n2. Testing with IMPROVED settings:")
    print("   - Paths: 5, Depth: 3, Model: gpt2 (optimized)")
    
    improved_runner = ToTRunner(
        model_name="gpt2",
        num_paths=5,
        max_depth=3
    )
    
    start_time = time.time()
    improved_result = improved_runner.run_single_task("geometry_001", save_results=False)
    improved_time = time.time() - start_time
    
    print(f"   Result: {improved_result.get('final_answer', 'No answer')}")
    print(f"   Consensus: {improved_result.get('consensus_score', 0):.3f}")
    print(f"   Time: {improved_time:.2f}s")
    
    # Compare results
    print("\n3. COMPARISON:")
    baseline_consensus = baseline_result.get('consensus_score', 0)
    improved_consensus = improved_result.get('consensus_score', 0)
    
    consensus_improvement = ((improved_consensus - baseline_consensus) / baseline_consensus * 100) if baseline_consensus > 0 else 0
    time_change = ((improved_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
    
    print(f"   Consensus Score: {baseline_consensus:.3f} → {improved_consensus:.3f} ({consensus_improvement:+.1f}%)")
    print(f"   Execution Time: {baseline_time:.2f}s → {improved_time:.2f}s ({time_change:+.1f}%)")
    
    return {
        "baseline": baseline_result,
        "improved": improved_result,
        "consensus_improvement": consensus_improvement,
        "time_change": time_change
    }


def test_domain_performance():
    """Test performance across different domains"""
    print("\n" + "="*80)
    print("TESTING DOMAIN-SPECIFIC PERFORMANCE")
    print("="*80)
    
    # Get available tasks by domain
    runner = ToTRunner(num_paths=5, max_depth=3)
    
    domains_to_test = ["geometry", "math", "logic", "probability"]
    domain_results = {}
    
    for domain in domains_to_test:
        print(f"\nTesting {domain.upper()} domain:")
        
        try:
            domain_tasks = runner.task_loader.get_tasks_by_domain(domain)
            if not domain_tasks:
                print(f"   No tasks found for {domain}")
                continue
                
            # Test first task in domain
            task_id = domain_tasks[0]["id"]
            print(f"   Task: {task_id}")
            
            start_time = time.time()
            result = runner.run_single_task(task_id, save_results=False)
            execution_time = time.time() - start_time
            
            domain_results[domain] = {
                "task_id": task_id,
                "result": result,
                "execution_time": execution_time
            }
            
            print(f"   Answer: {result.get('final_answer', 'No answer')[:50]}...")
            print(f"   Consensus: {result.get('consensus_score', 0):.3f}")
            print(f"   Time: {execution_time:.2f}s")
            
        except Exception as e:
            print(f"   Error testing {domain}: {e}")
            domain_results[domain] = {"error": str(e)}
    
    return domain_results


def test_model_recommendations():
    """Test model recommendation system"""
    print("\n" + "="*80)
    print("TESTING MODEL RECOMMENDATIONS")
    print("="*80)
    
    # Show system compatibility
    ModelConfig.print_system_info()
    
    # Test different VRAM scenarios
    vram_scenarios = [1.0, 2.0, 4.0, 8.0, 16.0]
    
    print("\nModel recommendations for different VRAM amounts:")
    for vram in vram_scenarios:
        recommended = ModelConfig.get_recommended_model(vram)
        model_info = ModelConfig.get_model_info(recommended)
        print(f"   {vram}GB VRAM → {recommended} ({model_info['reasoning_capability']} reasoning)")


def test_generation_parameters():
    """Test domain-specific generation parameters"""
    print("\n" + "="*80)
    print("TESTING GENERATION PARAMETERS")
    print("="*80)
    
    domains = ["math", "geometry", "logic", "probability", "code"]
    models = ["gpt2", "gpt2-medium", "distilgpt2"]
    
    print("\nOptimized parameters by domain and model:")
    for model in models:
        print(f"\n{model.upper()}:")
        for domain in domains:
            params = ModelConfig.get_optimized_generation_params(model, domain)
            print(f"   {domain:12}: temp={params['temperature']:.2f}, top_p={params['top_p']:.2f}")


def run_comprehensive_test():
    """Run all improvement tests"""
    print("PROMPT ENGINEERING PIPELINE - IMPROVEMENT VALIDATION")
    print("="*80)
    print("Testing enhanced reasoning, domain-specific prompts, and verification")
    
    results = {}
    
    try:
        # Test 1: Single task improvement
        results["single_task"] = test_single_task_improvement()
        
        # Test 2: Domain performance
        results["domain_performance"] = test_domain_performance()
        
        # Test 3: Model recommendations
        test_model_recommendations()
        
        # Test 4: Generation parameters
        test_generation_parameters()
        
        # Summary
        print("\n" + "="*80)
        print("IMPROVEMENT SUMMARY")
        print("="*80)
        
        if "single_task" in results:
            single_task = results["single_task"]
            print(f"Single Task Improvement:")
            print(f"   Consensus Score: {single_task['consensus_improvement']:+.1f}%")
            print(f"   Execution Time: {single_task['time_change']:+.1f}%")
        
        if "domain_performance" in results:
            domain_perf = results["domain_performance"]
            successful_domains = [d for d in domain_perf if "error" not in domain_perf[d]]
            print(f"\nDomain Testing:")
            print(f"   Successful domains: {len(successful_domains)}")
            print(f"   Average consensus: {sum(domain_perf[d]['result'].get('consensus_score', 0) for d in successful_domains) / len(successful_domains):.3f}" if successful_domains else "   No successful tests")
        
        print("\nKey Improvements Implemented:")
        print("   ✓ Increased reasoning paths (3→5) and depth (2→3)")
        print("   ✓ Domain-specific prompt templates")
        print("   ✓ Enhanced answer extraction with regex patterns")
        print("   ✓ Answer verification and consistency checking")
        print("   ✓ Optimized generation parameters per domain")
        print("   ✓ Model recommendation system")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during comprehensive test: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline improvements")
    parser.add_argument("--test", choices=["single", "domain", "models", "params", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "single":
        test_single_task_improvement()
    elif args.test == "domain":
        test_domain_performance()
    elif args.test == "models":
        test_model_recommendations()
    elif args.test == "params":
        test_generation_parameters()
    else:
        run_comprehensive_test()
