"""
Optimization Runner
Integrates prompt optimization with ToT reasoning for automated improvement
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from task_loader import TaskLoader
from reasoning.tot_engine import ToTEngine
from prompt_optimization.optimizer import PromptOptimizer
from prompt_optimization.prompt_manager import PromptManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationRunner:
    """Main runner for automated prompt optimization pipeline"""
    
    def __init__(self,
                 tasks_dir: str = "tasks",
                 prompts_dir: str = "prompts", 
                 logs_dir: str = "logs",
                 model_name: str = "gpt2",
                 max_iterations: int = 3,
                 min_improvement: float = 0.1):
        """
        Initialize Optimization Runner
        
        Args:
            tasks_dir: Directory containing task files
            prompts_dir: Directory for storing prompts
            logs_dir: Directory for storing logs
            model_name: HuggingFace model name
            max_iterations: Maximum optimization iterations
            min_improvement: Minimum improvement threshold
        """
        # Get project root directory
        self.project_root = Path(__file__).parent.parent.parent
        
        # Initialize components
        self.task_loader = TaskLoader(str(self.project_root / tasks_dir))
        self.prompt_manager = PromptManager(
            str(self.project_root / prompts_dir),
            str(self.project_root / logs_dir)
        )
        self.optimizer = PromptOptimizer(
            model_name=model_name,
            max_iterations=max_iterations,
            min_improvement_threshold=min_improvement
        )
        
        # Initialize ToT engine for evaluation (with fewer paths for speed)
        self.tot_engine = ToTEngine(
            model_name=model_name,
            num_paths=2,  # Fewer paths for faster optimization
            max_depth=2
        )
        
        logger.info("Optimization Runner initialized successfully")
    
    def optimize_task_prompt(self, task_id: str, use_existing: bool = True) -> dict:
        """
        Optimize the prompt for a specific task
        
        Args:
            task_id: Task identifier
            use_existing: Whether to use existing prompt as starting point
            
        Returns:
            Optimization result summary
        """
        logger.info(f"Starting prompt optimization for task: {task_id}")
        
        try:
            # Load task data
            task_data = self.task_loader.load_task(task_id)
            logger.info(f"Loaded task: {task_data['title']}")
            
            # Get initial prompt
            if use_existing:
                initial_prompt = self.prompt_manager.load_prompt_version(task_id)
                if initial_prompt is None:
                    logger.info("No existing prompt found, creating default")
                    initial_prompt = self.prompt_manager.create_initial_prompt(task_data)
                    # Save the initial prompt
                    self.prompt_manager.save_prompt_version(
                        task_id=task_id,
                        prompt_content=initial_prompt,
                        version=1,
                        performance_score=0.0,
                        metadata={"type": "initial_default"}
                    )
            else:
                initial_prompt = self.prompt_manager.create_initial_prompt(task_data)
            
            logger.info(f"Initial prompt: {initial_prompt[:100]}...")
            
            # Run optimization
            optimization_result = self.optimizer.optimize_prompt(
                task_data=task_data,
                initial_prompt=initial_prompt,
                tot_engine=self.tot_engine
            )
            
            # Save optimization results
            self.prompt_manager.save_optimization_result(optimization_result)
            
            # Create summary
            result_summary = {
                "task_id": task_id,
                "optimization_successful": optimization_result.improvement_score > 0,
                "improvement_score": optimization_result.improvement_score,
                "iterations": optimization_result.optimization_iterations,
                "execution_time": optimization_result.execution_time,
                "original_prompt": optimization_result.original_prompt,
                "best_prompt": optimization_result.best_prompt,
                "variants_generated": len(optimization_result.generated_variants)
            }
            
            logger.info(f"Optimization completed for {task_id}")
            logger.info(f"Improvement score: {optimization_result.improvement_score:.3f}")
            
            return result_summary
            
        except Exception as e:
            logger.error(f"Error optimizing task {task_id}: {e}")
            return {
                "task_id": task_id,
                "error": str(e),
                "optimization_successful": False
            }
    
    def optimize_multiple_tasks(self, task_ids: list = None) -> list:
        """
        Optimize prompts for multiple tasks
        
        Args:
            task_ids: List of task IDs (if None, optimizes all tasks)
            
        Returns:
            List of optimization result summaries
        """
        if task_ids is None:
            # Get all task IDs from registry
            registry = self.task_loader.load_registry()
            task_ids = [task["id"] for task in registry["task_registry"]["tasks"]]
        
        logger.info(f"Starting optimization for {len(task_ids)} tasks")
        
        results = []
        for task_id in task_ids:
            result = self.optimize_task_prompt(task_id)
            results.append(result)
        
        logger.info(f"Completed optimization for {len(results)} tasks")
        return results
    
    def optimize_domain_tasks(self, domain: str) -> list:
        """
        Optimize prompts for all tasks in a specific domain
        
        Args:
            domain: Domain name (e.g., "math", "logic")
            
        Returns:
            List of optimization result summaries
        """
        logger.info(f"Starting optimization for domain: {domain}")
        
        # Get tasks from domain
        domain_tasks = self.task_loader.get_tasks_by_domain(domain)
        task_ids = [task["id"] for task in domain_tasks]
        
        if not task_ids:
            logger.warning(f"No tasks found for domain: {domain}")
            return []
        
        return self.optimize_multiple_tasks(task_ids)
    
    def run_optimization_cycle(self, 
                              task_ids: list = None,
                              performance_threshold: float = 0.7) -> dict:
        """
        Run a complete optimization cycle
        
        Args:
            task_ids: List of task IDs to optimize
            performance_threshold: Minimum performance threshold
            
        Returns:
            Cycle summary report
        """
        logger.info("Starting optimization cycle")
        
        # Run optimizations
        optimization_results = self.optimize_multiple_tasks(task_ids)
        
        # Analyze results
        successful_optimizations = [r for r in optimization_results if r.get("optimization_successful", False)]
        failed_optimizations = [r for r in optimization_results if not r.get("optimization_successful", False)]
        
        # Calculate statistics
        total_tasks = len(optimization_results)
        success_rate = len(successful_optimizations) / total_tasks if total_tasks > 0 else 0
        
        avg_improvement = sum(r.get("improvement_score", 0) for r in successful_optimizations) / len(successful_optimizations) if successful_optimizations else 0
        avg_execution_time = sum(r.get("execution_time", 0) for r in optimization_results) / total_tasks if total_tasks > 0 else 0
        
        # Identify tasks needing further optimization
        low_performance_tasks = [
            r for r in successful_optimizations 
            if r.get("improvement_score", 0) < performance_threshold
        ]
        
        # Create cycle summary
        cycle_summary = {
            "cycle_completed_at": logger.handlers[0].formatter.formatTime(logger.makeRecord("", 0, "", 0, "", (), None)),
            "total_tasks": total_tasks,
            "successful_optimizations": len(successful_optimizations),
            "failed_optimizations": len(failed_optimizations),
            "success_rate": success_rate,
            "average_improvement": avg_improvement,
            "average_execution_time": avg_execution_time,
            "tasks_needing_further_optimization": len(low_performance_tasks),
            "results": optimization_results
        }
        
        logger.info(f"Optimization cycle completed")
        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Average improvement: {avg_improvement:.3f}")
        logger.info(f"Tasks needing further work: {len(low_performance_tasks)}")
        
        return cycle_summary
    
    def display_optimization_results(self, results: list):
        """
        Display optimization results in a formatted way
        
        Args:
            results: List of optimization result summaries
        """
        print("\n" + "="*80)
        print("PROMPT OPTIMIZATION RESULTS")
        print("="*80)
        
        for result in results:
            if "error" in result:
                print(f"\n❌ Task: {result['task_id']}")
                print(f"   Status: FAILED")
                print(f"   Error: {result['error']}")
            else:
                status = "✅ IMPROVED" if result.get("optimization_successful", False) else "⚠️  NO IMPROVEMENT"
                print(f"\n{status} Task: {result['task_id']}")
                print(f"   Improvement Score: {result.get('improvement_score', 0):.3f}")
                print(f"   Iterations: {result.get('iterations', 0)}")
                print(f"   Execution Time: {result.get('execution_time', 0):.2f}s")
                print(f"   Variants Generated: {result.get('variants_generated', 0)}")
                
                if result.get("optimization_successful", False):
                    print(f"   Original: {result.get('original_prompt', '')[:60]}...")
                    print(f"   Optimized: {result.get('best_prompt', '')[:60]}...")
        
        print("\n" + "="*80)
    
    def get_optimization_summary(self) -> dict:
        """Get summary of all optimization activities"""
        prompt_summary = self.prompt_manager.get_best_prompts_summary()
        
        summary = {
            "total_tasks_with_prompts": len(prompt_summary),
            "tasks": prompt_summary,
            "optimization_logs_available": len(list(self.prompt_manager.optimization_logs_dir.glob("*.json")))
        }
        
        return summary


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated prompt optimization")
    parser.add_argument("--task", type=str, help="Specific task ID to optimize")
    parser.add_argument("--domain", type=str, help="Domain to optimize (e.g., math, logic)")
    parser.add_argument("--all", action="store_true", help="Optimize all tasks")
    parser.add_argument("--cycle", action="store_true", help="Run complete optimization cycle")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--iterations", type=int, default=3, help="Max optimization iterations (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Min improvement threshold (default: 0.1)")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = OptimizationRunner(
        model_name=args.model,
        max_iterations=args.iterations,
        min_improvement=args.threshold
    )
    
    # Determine what to run
    results = []
    
    if args.task:
        results = [runner.optimize_task_prompt(args.task)]
    elif args.domain:
        results = runner.optimize_domain_tasks(args.domain)
    elif args.all or args.cycle:
        if args.cycle:
            cycle_result = runner.run_optimization_cycle()
            results = cycle_result["results"]
        else:
            results = runner.optimize_multiple_tasks()
    else:
        print("Please specify --task, --domain, --all, or --cycle")
        return
    
    # Display results
    runner.display_optimization_results(results)
    
    # Show summary
    if results:
        successful = len([r for r in results if r.get("optimization_successful", False)])
        total = len(results)
        avg_improvement = sum(r.get("improvement_score", 0) for r in results) / total
        
        print(f"\nSummary: {successful}/{total} tasks improved")
        print(f"Average improvement: {avg_improvement:.3f}")


if __name__ == "__main__":
    main()
