"""
Evaluation Runner
Integrates evaluation and reflection systems for comprehensive pipeline assessment
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from task_loader import TaskLoader
from reasoning.tot_engine import ToTEngine
from reasoning.result_logger import ResultLogger
from evaluation.evaluator import Evaluator
from evaluation.reflection import ReflectionSystem
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Main runner for comprehensive pipeline evaluation and reflection"""
    
    def __init__(self,
                 tasks_dir: str = "tasks",
                 logs_dir: str = "logs",
                 evaluation_dir: str = "evaluation",
                 model_name: str = "gpt2"):
        """
        Initialize Evaluation Runner
        
        Args:
            tasks_dir: Directory containing task files
            logs_dir: Directory for storing logs
            evaluation_dir: Directory for evaluation reports
            model_name: HuggingFace model name
        """
        # Get project root directory
        self.project_root = Path(__file__).parent.parent.parent
        
        # Initialize components
        self.task_loader = TaskLoader(str(self.project_root / tasks_dir))
        self.tot_engine = ToTEngine(model_name=model_name, num_paths=2, max_depth=2)
        self.result_logger = ResultLogger(str(self.project_root / logs_dir))
        self.evaluator = Evaluator(str(self.project_root / evaluation_dir))
        self.reflection_system = ReflectionSystem(str(self.project_root / evaluation_dir))
        
        logger.info("Evaluation Runner initialized successfully")
    
    def run_task_evaluation(self, task_id: str, prompt_version: int = 1) -> dict:
        """
        Run complete evaluation for a single task
        
        Args:
            task_id: Task identifier
            prompt_version: Version of prompt used
            
        Returns:
            Evaluation result summary
        """
        logger.info(f"Running evaluation for task: {task_id}")
        
        try:
            # Load task
            task_data = self.task_loader.load_task(task_id)
            logger.info(f"Loaded task: {task_data['title']}")
            
            # Run ToT reasoning
            tot_result = self.tot_engine.reason(task_data)
            
            # Evaluate the result
            evaluation = self.evaluator.evaluate_task_result(
                task_data, tot_result, prompt_version
            )
            
            # Create summary
            result_summary = {
                "task_id": task_id,
                "evaluation_successful": True,
                "accuracy_score": evaluation.accuracy_score,
                "reasoning_coherence": evaluation.reasoning_coherence,
                "hallucination_rate": evaluation.hallucination_rate,
                "execution_time": evaluation.execution_time,
                "consensus_score": evaluation.consensus_score,
                "expected_answer": evaluation.expected_answer,
                "generated_answer": evaluation.generated_answer,
                "prompt_version": prompt_version
            }
            
            logger.info(f"Evaluation completed for {task_id}")
            logger.info(f"Accuracy: {evaluation.accuracy_score:.3f}")
            
            return result_summary, evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating task {task_id}: {e}")
            return {
                "task_id": task_id,
                "error": str(e),
                "evaluation_successful": False
            }, None
    
    def run_multiple_evaluations(self, task_ids: list = None) -> tuple:
        """
        Run evaluations for multiple tasks
        
        Args:
            task_ids: List of task IDs (if None, evaluates all tasks)
            
        Returns:
            Tuple of (result_summaries, evaluations)
        """
        if task_ids is None:
            # Get all task IDs from registry
            registry = self.task_loader.load_registry()
            task_ids = [task["id"] for task in registry["task_registry"]["tasks"]]
        
        logger.info(f"Running evaluations for {len(task_ids)} tasks")
        
        result_summaries = []
        evaluations = []
        
        for task_id in task_ids:
            result_summary, evaluation = self.run_task_evaluation(task_id)
            result_summaries.append(result_summary)
            if evaluation:
                evaluations.append(evaluation)
        
        logger.info(f"Completed evaluations for {len(result_summaries)} tasks")
        return result_summaries, evaluations
    
    def run_domain_evaluation(self, domain: str) -> tuple:
        """
        Run evaluations for all tasks in a specific domain
        
        Args:
            domain: Domain name (e.g., "math", "logic")
            
        Returns:
            Tuple of (result_summaries, evaluations)
        """
        logger.info(f"Running evaluation for domain: {domain}")
        
        # Get tasks from domain
        domain_tasks = self.task_loader.get_tasks_by_domain(domain)
        task_ids = [task["id"] for task in domain_tasks]
        
        if not task_ids:
            logger.warning(f"No tasks found for domain: {domain}")
            return [], []
        
        return self.run_multiple_evaluations(task_ids)
    
    def generate_evaluation_report(self, evaluations: list, save_report: bool = True) -> dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluations: List of TaskEvaluation objects
            save_report: Whether to save report to file
            
        Returns:
            Evaluation report dictionary
        """
        logger.info("Generating evaluation report")
        
        if not evaluations:
            logger.warning("No evaluations provided for report generation")
            return {}
        
        # Generate report
        report = self.evaluator.generate_evaluation_report(evaluations)
        
        # Save report if requested
        if save_report:
            self.evaluator.save_evaluation_report(report)
        
        # Convert to dictionary for return
        report_summary = {
            "report_id": report.report_id,
            "total_tasks": report.total_tasks,
            "overall_accuracy": report.aggregate_metrics.get("overall_accuracy", 0),
            "overall_coherence": report.aggregate_metrics.get("overall_coherence", 0),
            "overall_hallucination_rate": report.aggregate_metrics.get("overall_hallucination_rate", 0),
            "average_execution_time": report.aggregate_metrics.get("average_execution_time", 0),
            "domain_breakdown": report.domain_breakdown,
            "recommendations": report.recommendations
        }
        
        logger.info(f"Evaluation report generated: {report.report_id}")
        return report_summary
    
    def run_reflection_analysis(self, report_ids: list = None) -> dict:
        """
        Run reflection analysis on evaluation reports
        
        Args:
            report_ids: Specific report IDs to analyze (if None, analyzes all)
            
        Returns:
            Reflection report summary
        """
        logger.info("Running reflection analysis")
        
        # Generate reflection report
        reflection_report = self.reflection_system.generate_reflection_report(report_ids)
        
        # Save reflection report
        self.reflection_system.save_reflection_report(reflection_report)
        
        # Create summary
        reflection_summary = {
            "report_id": reflection_report.report_id,
            "reports_analyzed": len(reflection_report.evaluation_reports_analyzed),
            "key_insights_count": len(reflection_report.key_insights),
            "optimization_opportunities": reflection_report.optimization_opportunities,
            "next_steps": reflection_report.next_steps[:5],  # Top 5 next steps
            "insights_summary": [
                {
                    "category": insight.category,
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence
                }
                for insight in reflection_report.key_insights
            ]
        }
        
        logger.info(f"Reflection analysis completed: {reflection_report.report_id}")
        return reflection_summary
    
    def run_complete_evaluation_cycle(self, task_ids: list = None) -> dict:
        """
        Run complete evaluation cycle: tasks -> evaluation -> reflection
        
        Args:
            task_ids: List of task IDs to evaluate
            
        Returns:
            Complete cycle summary
        """
        logger.info("Starting complete evaluation cycle")
        
        # Step 1: Run task evaluations
        result_summaries, evaluations = self.run_multiple_evaluations(task_ids)
        
        # Step 2: Generate evaluation report
        evaluation_report = self.generate_evaluation_report(evaluations, save_report=True)
        
        # Step 3: Run reflection analysis
        reflection_report = self.run_reflection_analysis()
        
        # Create cycle summary
        cycle_summary = {
            "cycle_completed_at": logger.handlers[0].formatter.formatTime(logger.makeRecord("", 0, "", 0, "", (), None)),
            "tasks_evaluated": len(result_summaries),
            "successful_evaluations": len([r for r in result_summaries if r.get("evaluation_successful", False)]),
            "evaluation_report": evaluation_report,
            "reflection_report": reflection_report,
            "key_findings": {
                "overall_accuracy": evaluation_report.get("overall_accuracy", 0),
                "top_recommendation": evaluation_report.get("recommendations", ["None"])[0] if evaluation_report.get("recommendations") else "None",
                "primary_insight": reflection_report.get("insights_summary", [{}])[0].get("description", "None") if reflection_report.get("insights_summary") else "None",
                "next_priority": reflection_report.get("next_steps", ["None"])[0] if reflection_report.get("next_steps") else "None"
            }
        }
        
        logger.info("Complete evaluation cycle finished")
        return cycle_summary
    
    def display_evaluation_results(self, result_summaries: list):
        """
        Display evaluation results in a formatted way
        
        Args:
            result_summaries: List of evaluation result summaries
        """
        print("\n" + "="*80)
        print("PIPELINE EVALUATION RESULTS")
        print("="*80)
        
        successful_evals = [r for r in result_summaries if r.get("evaluation_successful", False)]
        failed_evals = [r for r in result_summaries if not r.get("evaluation_successful", False)]
        
        print(f"\nSummary: {len(successful_evals)}/{len(result_summaries)} evaluations successful")
        
        if successful_evals:
            avg_accuracy = sum(r.get("accuracy_score", 0) for r in successful_evals) / len(successful_evals)
            avg_coherence = sum(r.get("reasoning_coherence", 0) for r in successful_evals) / len(successful_evals)
            avg_hallucination = sum(r.get("hallucination_rate", 0) for r in successful_evals) / len(successful_evals)
            
            print(f"Average Accuracy: {avg_accuracy:.1%}")
            print(f"Average Coherence: {avg_coherence:.1%}")
            print(f"Average Hallucination Rate: {avg_hallucination:.1%}")
        
        print("\nDetailed Results:")
        for result in result_summaries:
            if "error" in result:
                print(f"\n❌ Task: {result['task_id']}")
                print(f"   Status: FAILED")
                print(f"   Error: {result['error']}")
            else:
                accuracy_icon = "✅" if result.get("accuracy_score", 0) >= 0.7 else "⚠️" if result.get("accuracy_score", 0) >= 0.3 else "❌"
                print(f"\n{accuracy_icon} Task: {result['task_id']}")
                print(f"   Accuracy: {result.get('accuracy_score', 0):.1%}")
                print(f"   Coherence: {result.get('reasoning_coherence', 0):.1%}")
                print(f"   Hallucination: {result.get('hallucination_rate', 0):.1%}")
                print(f"   Time: {result.get('execution_time', 0):.2f}s")
                print(f"   Expected: {result.get('expected_answer', '')[:40]}...")
                print(f"   Generated: {result.get('generated_answer', '')[:40]}...")
        
        print("\n" + "="*80)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive pipeline evaluation")
    parser.add_argument("--task", type=str, help="Specific task ID to evaluate")
    parser.add_argument("--domain", type=str, help="Domain to evaluate (e.g., math, logic)")
    parser.add_argument("--all", action="store_true", help="Evaluate all tasks")
    parser.add_argument("--cycle", action="store_true", help="Run complete evaluation cycle")
    parser.add_argument("--reflect", action="store_true", help="Run reflection analysis only")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (default: gpt2)")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(model_name=args.model)
    
    # Determine what to run
    if args.reflect:
        # Run reflection analysis only
        reflection_summary = runner.run_reflection_analysis()
        print("\nReflection Analysis Results:")
        print(f"Reports analyzed: {reflection_summary['reports_analyzed']}")
        print(f"Key insights: {reflection_summary['key_insights_count']}")
        print("\nTop insights:")
        for insight in reflection_summary['insights_summary'][:3]:
            print(f"- {insight['description']} (confidence: {insight['confidence']:.1%})")
        print("\nNext steps:")
        for step in reflection_summary['next_steps']:
            print(f"- {step}")
        
    elif args.cycle:
        # Run complete evaluation cycle
        cycle_summary = runner.run_complete_evaluation_cycle()
        print("\nComplete Evaluation Cycle Results:")
        print(f"Tasks evaluated: {cycle_summary['tasks_evaluated']}")
        print(f"Overall accuracy: {cycle_summary['key_findings']['overall_accuracy']:.1%}")
        print(f"Top recommendation: {cycle_summary['key_findings']['top_recommendation']}")
        print(f"Primary insight: {cycle_summary['key_findings']['primary_insight']}")
        print(f"Next priority: {cycle_summary['key_findings']['next_priority']}")
        
    else:
        # Run task evaluations
        if args.task:
            result_summary, evaluation = runner.run_task_evaluation(args.task)
            result_summaries = [result_summary]
            evaluations = [evaluation] if evaluation else []
        elif args.domain:
            result_summaries, evaluations = runner.run_domain_evaluation(args.domain)
        elif args.all:
            result_summaries, evaluations = runner.run_multiple_evaluations()
        else:
            print("Please specify --task, --domain, --all, --cycle, or --reflect")
            return
        
        # Display results
        runner.display_evaluation_results(result_summaries)
        
        # Generate report if we have evaluations
        if evaluations:
            report_summary = runner.generate_evaluation_report(evaluations)
            print(f"\nEvaluation report generated: {report_summary.get('report_id', 'unknown')}")


if __name__ == "__main__":
    main()
