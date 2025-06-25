"""
ToT Reasoning Runner
Main script to run Tree-of-Thought reasoning on tasks
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from task_loader import TaskLoader
from reasoning.tot_engine import ToTEngine
from reasoning.result_logger import ResultLogger
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ToTRunner:
    """Main runner for Tree-of-Thought reasoning pipeline"""
    
    def __init__(self,
                 tasks_dir: str = "tasks",
                 logs_dir: str = "logs",
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 num_paths: int = 5,
                 max_depth: int = 3):
        """
        Initialize ToT Runner
        
        Args:
            tasks_dir: Directory containing task files
            logs_dir: Directory for storing logs
            model_name: HuggingFace model name
            num_paths: Number of reasoning paths to generate
            max_depth: Maximum depth for reasoning tree
        """
        # Get project root directory
        self.project_root = Path(__file__).parent.parent.parent
        
        # Initialize components
        self.task_loader = TaskLoader(str(self.project_root / tasks_dir))
        self.tot_engine = ToTEngine(model_name=model_name, num_paths=num_paths, max_depth=max_depth)
        self.result_logger = ResultLogger(str(self.project_root / logs_dir))
        
        logger.info("ToT Runner initialized successfully")
        
    def run_single_task(self, task_id: str, save_results: bool = True) -> dict:
        """
        Run ToT reasoning on a single task
        
        Args:
            task_id: Task identifier
            save_results: Whether to save results to file
            
        Returns:
            Result dictionary
        """
        logger.info(f"Running ToT reasoning for task: {task_id}")
        
        try:
            # Load task
            task_data = self.task_loader.load_task(task_id)
            logger.info(f"Loaded task: {task_data['title']}")
            
            # Run reasoning
            result = self.tot_engine.reason(task_data)
            
            # Save results if requested
            if save_results:
                self.result_logger.save_result(result)
                
            # Create summary for return
            result_summary = {
                "task_id": result.task_id,
                "final_answer": result.final_answer,
                "expected_answer": task_data.get("expected_solution", "Unknown"),
                "consensus_score": result.consensus_score,
                "execution_time": result.execution_time,
                "num_paths": len(result.reasoning_paths),
                "model_info": result.model_info
            }
            
            logger.info(f"Task completed successfully: {task_id}")
            return result_summary
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            return {
                "task_id": task_id,
                "error": str(e),
                "status": "failed"
            }
    
    def run_multiple_tasks(self, task_ids: list = None, save_results: bool = True) -> list:
        """
        Run ToT reasoning on multiple tasks
        
        Args:
            task_ids: List of task IDs (if None, runs all tasks)
            save_results: Whether to save results to files
            
        Returns:
            List of result summaries
        """
        if task_ids is None:
            # Get all task IDs from registry
            registry = self.task_loader.load_registry()
            task_ids = [task["id"] for task in registry["task_registry"]["tasks"]]
            
        logger.info(f"Running ToT reasoning on {len(task_ids)} tasks")
        
        results = []
        for task_id in task_ids:
            result = self.run_single_task(task_id, save_results)
            results.append(result)
            
        logger.info(f"Completed processing {len(results)} tasks")
        return results
    
    def run_domain_tasks(self, domain: str, save_results: bool = True) -> list:
        """
        Run ToT reasoning on all tasks from a specific domain
        
        Args:
            domain: Domain name (e.g., "math", "logic")
            save_results: Whether to save results to files
            
        Returns:
            List of result summaries
        """
        logger.info(f"Running ToT reasoning for domain: {domain}")
        
        # Get tasks from domain
        domain_tasks = self.task_loader.get_tasks_by_domain(domain)
        task_ids = [task["id"] for task in domain_tasks]
        
        if not task_ids:
            logger.warning(f"No tasks found for domain: {domain}")
            return []
            
        return self.run_multiple_tasks(task_ids, save_results)
    
    def generate_report(self, results: list, save_report: bool = True) -> dict:
        """
        Generate a summary report from results
        
        Args:
            results: List of result summaries
            save_report: Whether to save report to file
            
        Returns:
            Report dictionary
        """
        logger.info("Generating summary report")
        
        # Filter successful results
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        report = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "average_execution_time": sum(r.get("execution_time", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            "average_consensus_score": sum(r.get("consensus_score", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            "results": results
        }
        
        if save_report:
            self.result_logger.save_summary_report(report)
            
        logger.info(f"Report generated: {report['successful_tasks']}/{report['total_tasks']} tasks successful")
        return report
    
    def display_results(self, results: list):
        """
        Display results in a formatted way
        
        Args:
            results: List of result summaries
        """
        print("\n" + "="*80)
        print("TREE-OF-THOUGHT REASONING RESULTS")
        print("="*80)
        
        for result in results:
            if "error" in result:
                print(f"\n❌ Task: {result['task_id']}")
                print(f"   Status: FAILED")
                print(f"   Error: {result['error']}")
            else:
                print(f"\n✅ Task: {result['task_id']}")
                print(f"   Final Answer: {result['final_answer']}")
                print(f"   Expected Answer: {result['expected_answer']}")
                print(f"   Consensus Score: {result['consensus_score']:.3f}")
                print(f"   Execution Time: {result['execution_time']:.2f}s")
                print(f"   Reasoning Paths: {result['num_paths']}")
        
        print("\n" + "="*80)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Tree-of-Thought reasoning on tasks")
    parser.add_argument("--task", type=str, help="Specific task ID to run")
    parser.add_argument("--domain", type=str, help="Domain to run (e.g., math, logic)")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name (default: Mistral 7B Instruct)")
    parser.add_argument("--paths", type=int, default=5, help="Number of reasoning paths (default: 5)")
    parser.add_argument("--depth", type=int, default=3, help="Maximum reasoning depth (default: 3)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ToTRunner(
        model_name=args.model,
        num_paths=args.paths,
        max_depth=args.depth
    )
    
    # Determine what to run
    results = []
    
    if args.task:
        results = [runner.run_single_task(args.task, not args.no_save)]
    elif args.domain:
        results = runner.run_domain_tasks(args.domain, not args.no_save)
    elif args.all:
        results = runner.run_multiple_tasks(save_results=not args.no_save)
    else:
        print("Please specify --task, --domain, or --all")
        return
    
    # Display results
    runner.display_results(results)
    
    # Generate report
    if results:
        report = runner.generate_report(results, not args.no_save)
        print(f"\nSummary: {report['successful_tasks']}/{report['total_tasks']} tasks completed successfully")
        print(f"Average execution time: {report['average_execution_time']:.2f}s")
        print(f"Average consensus score: {report['average_consensus_score']:.3f}")


if __name__ == "__main__":
    main()
