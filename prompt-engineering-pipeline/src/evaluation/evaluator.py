"""
Evaluation Framework
Comprehensive evaluation system for prompt engineering pipeline performance
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskEvaluation:
    """Evaluation results for a single task"""
    task_id: str
    task_domain: str
    task_difficulty: str
    expected_answer: str
    generated_answer: str
    accuracy_score: float
    reasoning_coherence: float
    hallucination_rate: float
    execution_time: float
    consensus_score: float
    prompt_version: int
    evaluation_timestamp: str
    detailed_metrics: Dict[str, Any]


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    report_id: str
    evaluation_timestamp: str
    total_tasks: int
    task_evaluations: List[TaskEvaluation]
    aggregate_metrics: Dict[str, float]
    domain_breakdown: Dict[str, Dict[str, float]]
    difficulty_breakdown: Dict[str, Dict[str, float]]
    improvement_analysis: Dict[str, Any]
    recommendations: List[str]


class Evaluator:
    """Main evaluation system for the prompt engineering pipeline"""
    
    def __init__(self, evaluation_dir: str = "evaluation"):
        """
        Initialize Evaluator
        
        Args:
            evaluation_dir: Directory for storing evaluation results
        """
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluation criteria
        self.accuracy_evaluators = {
            "exact_match": self._exact_match_accuracy,
            "numerical_equivalence": self._numerical_equivalence_accuracy,
            "logical_equivalence": self._logical_equivalence_accuracy,
            "semantic_match": self._semantic_match_accuracy
        }
        
        logger.info("Evaluator initialized successfully")
    
    def evaluate_task_result(self, 
                           task_data: Dict[str, Any],
                           tot_result: Any,
                           prompt_version: int = 1) -> TaskEvaluation:
        """
        Evaluate the result of a single task
        
        Args:
            task_data: Original task data
            tot_result: ToT reasoning result
            prompt_version: Version of prompt used
            
        Returns:
            TaskEvaluation object
        """
        logger.info(f"Evaluating task: {task_data['id']}")
        
        # Extract basic information
        task_id = task_data["id"]
        task_domain = task_data.get("domain", "unknown")
        task_difficulty = task_data.get("difficulty", "unknown")
        expected_answer = task_data.get("expected_solution", "")
        generated_answer = tot_result.final_answer
        
        # Calculate accuracy score
        evaluation_rule = task_data.get("evaluation_rule", "exact_match")
        accuracy_score = self._calculate_accuracy(
            expected_answer, generated_answer, evaluation_rule
        )
        
        # Calculate reasoning coherence
        reasoning_coherence = self._calculate_reasoning_coherence(
            tot_result.reasoning_paths, task_data
        )
        
        # Calculate hallucination rate
        hallucination_rate = self._calculate_hallucination_rate(
            tot_result.reasoning_paths, task_data
        )
        
        # Gather detailed metrics
        detailed_metrics = {
            "num_reasoning_paths": len(tot_result.reasoning_paths),
            "avg_path_quality": sum(path.reasoning_quality for path in tot_result.reasoning_paths) / len(tot_result.reasoning_paths) if tot_result.reasoning_paths else 0,
            "model_info": tot_result.model_info,
            "answer_length": len(generated_answer),
            "reasoning_depth": max(len(path.nodes) for path in tot_result.reasoning_paths) if tot_result.reasoning_paths else 0
        }
        
        # Create evaluation
        evaluation = TaskEvaluation(
            task_id=task_id,
            task_domain=task_domain,
            task_difficulty=task_difficulty,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            accuracy_score=accuracy_score,
            reasoning_coherence=reasoning_coherence,
            hallucination_rate=hallucination_rate,
            execution_time=tot_result.execution_time,
            consensus_score=tot_result.consensus_score,
            prompt_version=prompt_version,
            evaluation_timestamp=datetime.now().isoformat(),
            detailed_metrics=detailed_metrics
        )
        
        logger.info(f"Task evaluation completed: {task_id}")
        logger.info(f"Accuracy: {accuracy_score:.3f}, Coherence: {reasoning_coherence:.3f}")
        
        return evaluation
    
    def _calculate_accuracy(self, expected: str, generated: str, rule: str) -> float:
        """Calculate accuracy score based on evaluation rule"""
        if rule in self.accuracy_evaluators:
            return self.accuracy_evaluators[rule](expected, generated)
        else:
            logger.warning(f"Unknown evaluation rule: {rule}, using exact_match")
            return self._exact_match_accuracy(expected, generated)
    
    def _exact_match_accuracy(self, expected: str, generated: str) -> float:
        """Exact string match accuracy"""
        expected_clean = expected.strip().lower()
        generated_clean = generated.strip().lower()
        return 1.0 if expected_clean == generated_clean else 0.0
    
    def _numerical_equivalence_accuracy(self, expected: str, generated: str) -> float:
        """Numerical equivalence accuracy"""
        # Extract numbers from both strings
        expected_numbers = re.findall(r'-?\d+\.?\d*', expected)
        generated_numbers = re.findall(r'-?\d+\.?\d*', generated)
        
        if not expected_numbers or not generated_numbers:
            return self._exact_match_accuracy(expected, generated)
        
        try:
            # Compare the first number found
            expected_num = float(expected_numbers[0])
            generated_num = float(generated_numbers[0])
            
            # Allow small floating point differences
            tolerance = 0.01
            return 1.0 if abs(expected_num - generated_num) <= tolerance else 0.0
            
        except ValueError:
            return 0.0
    
    def _logical_equivalence_accuracy(self, expected: str, generated: str) -> float:
        """Logical equivalence accuracy (simplified)"""
        # Simple keyword-based logical equivalence
        expected_keywords = set(re.findall(r'\b\w+\b', expected.lower()))
        generated_keywords = set(re.findall(r'\b\w+\b', generated.lower()))
        
        if not expected_keywords:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(expected_keywords.intersection(generated_keywords))
        union = len(expected_keywords.union(generated_keywords))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Consider it accurate if similarity is high
        return 1.0 if similarity >= 0.7 else similarity
    
    def _semantic_match_accuracy(self, expected: str, generated: str) -> float:
        """Semantic match accuracy (simplified)"""
        # Simple semantic matching based on key concepts
        expected_lower = expected.lower()
        generated_lower = generated.lower()
        
        # Check for key semantic indicators
        if "error" in expected_lower or "bug" in expected_lower:
            # For debugging tasks, check if the fix is mentioned
            if any(word in generated_lower for word in ["fix", "change", "correct", "error", "bug"]):
                return 0.8
        
        # Fallback to logical equivalence
        return self._logical_equivalence_accuracy(expected, generated)
    
    def _calculate_reasoning_coherence(self, reasoning_paths: List[Any], task_data: Dict[str, Any]) -> float:
        """Calculate reasoning coherence score"""
        if not reasoning_paths:
            return 0.0
        
        coherence_scores = []
        
        for path in reasoning_paths:
            path_coherence = 0.0
            
            # Check for logical progression
            if len(path.nodes) > 1:
                path_coherence += 0.3
            
            # Check for domain-relevant reasoning
            domain = task_data.get("domain", "")
            domain_keywords = {
                "math": ["calculate", "equation", "formula", "number", "result"],
                "logic": ["assume", "therefore", "if", "then", "conclusion"],
                "geometry": ["area", "triangle", "angle", "formula"],
                "probability": ["probability", "chance", "outcome", "calculate"]
            }
            
            if domain in domain_keywords:
                path_text = " ".join(node.content.lower() for node in path.nodes)
                keyword_matches = sum(1 for keyword in domain_keywords[domain] if keyword in path_text)
                path_coherence += min(0.4, keyword_matches * 0.1)
            
            # Check for step-by-step reasoning
            step_indicators = ["step", "first", "second", "then", "next", "finally"]
            path_text = " ".join(node.content.lower() for node in path.nodes)
            step_count = sum(1 for indicator in step_indicators if indicator in path_text)
            path_coherence += min(0.3, step_count * 0.1)
            
            coherence_scores.append(min(1.0, path_coherence))
        
        return statistics.mean(coherence_scores)
    
    def _calculate_hallucination_rate(self, reasoning_paths: List[Any], task_data: Dict[str, Any]) -> float:
        """Calculate hallucination rate (simplified heuristic)"""
        if not reasoning_paths:
            return 1.0  # No reasoning is considered high hallucination
        
        hallucination_indicators = [
            "i don't know", "unclear", "cannot determine", "impossible",
            "random", "guess", "maybe", "probably not", "unsure"
        ]
        
        total_nodes = 0
        hallucinated_nodes = 0
        
        for path in reasoning_paths:
            for node in path.nodes:
                total_nodes += 1
                node_text = node.content.lower()
                
                # Check for hallucination indicators
                if any(indicator in node_text for indicator in hallucination_indicators):
                    hallucinated_nodes += 1
                
                # Check for repetitive content (sign of hallucination)
                words = node_text.split()
                if len(words) > 5:
                    unique_words = set(words)
                    if len(unique_words) / len(words) < 0.5:  # Less than 50% unique words
                        hallucinated_nodes += 1
        
        return hallucinated_nodes / total_nodes if total_nodes > 0 else 0.0
    
    def evaluate_multiple_tasks(self, 
                               task_results: List[Tuple[Dict[str, Any], Any, int]]) -> List[TaskEvaluation]:
        """
        Evaluate multiple task results
        
        Args:
            task_results: List of (task_data, tot_result, prompt_version) tuples
            
        Returns:
            List of TaskEvaluation objects
        """
        logger.info(f"Evaluating {len(task_results)} tasks")
        
        evaluations = []
        for task_data, tot_result, prompt_version in task_results:
            evaluation = self.evaluate_task_result(task_data, tot_result, prompt_version)
            evaluations.append(evaluation)
        
        logger.info(f"Completed evaluation of {len(evaluations)} tasks")
        return evaluations
    
    def generate_evaluation_report(self, evaluations: List[TaskEvaluation]) -> EvaluationReport:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluations: List of task evaluations
            
        Returns:
            EvaluationReport object
        """
        logger.info("Generating evaluation report")
        
        if not evaluations:
            logger.warning("No evaluations provided for report generation")
            return EvaluationReport(
                report_id="empty_report",
                evaluation_timestamp=datetime.now().isoformat(),
                total_tasks=0,
                task_evaluations=[],
                aggregate_metrics={},
                domain_breakdown={},
                difficulty_breakdown={},
                improvement_analysis={},
                recommendations=["No tasks evaluated"]
            )
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            "overall_accuracy": statistics.mean([e.accuracy_score for e in evaluations]),
            "overall_coherence": statistics.mean([e.reasoning_coherence for e in evaluations]),
            "overall_hallucination_rate": statistics.mean([e.hallucination_rate for e in evaluations]),
            "average_execution_time": statistics.mean([e.execution_time for e in evaluations]),
            "average_consensus_score": statistics.mean([e.consensus_score for e in evaluations]),
            "accuracy_std": statistics.stdev([e.accuracy_score for e in evaluations]) if len(evaluations) > 1 else 0.0
        }
        
        # Domain breakdown
        domain_breakdown = self._calculate_domain_breakdown(evaluations)
        
        # Difficulty breakdown
        difficulty_breakdown = self._calculate_difficulty_breakdown(evaluations)
        
        # Improvement analysis
        improvement_analysis = self._analyze_improvements(evaluations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(evaluations, aggregate_metrics)
        
        # Create report
        report = EvaluationReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            evaluation_timestamp=datetime.now().isoformat(),
            total_tasks=len(evaluations),
            task_evaluations=evaluations,
            aggregate_metrics=aggregate_metrics,
            domain_breakdown=domain_breakdown,
            difficulty_breakdown=difficulty_breakdown,
            improvement_analysis=improvement_analysis,
            recommendations=recommendations
        )
        
        logger.info("Evaluation report generated successfully")
        return report
    
    def _calculate_domain_breakdown(self, evaluations: List[TaskEvaluation]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics breakdown by domain"""
        domain_groups = {}
        
        for eval in evaluations:
            domain = eval.task_domain
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(eval)
        
        breakdown = {}
        for domain, evals in domain_groups.items():
            breakdown[domain] = {
                "count": len(evals),
                "accuracy": statistics.mean([e.accuracy_score for e in evals]),
                "coherence": statistics.mean([e.reasoning_coherence for e in evals]),
                "hallucination_rate": statistics.mean([e.hallucination_rate for e in evals]),
                "execution_time": statistics.mean([e.execution_time for e in evals])
            }
        
        return breakdown
    
    def _calculate_difficulty_breakdown(self, evaluations: List[TaskEvaluation]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics breakdown by difficulty"""
        difficulty_groups = {}
        
        for eval in evaluations:
            difficulty = eval.task_difficulty
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(eval)
        
        breakdown = {}
        for difficulty, evals in difficulty_groups.items():
            breakdown[difficulty] = {
                "count": len(evals),
                "accuracy": statistics.mean([e.accuracy_score for e in evals]),
                "coherence": statistics.mean([e.reasoning_coherence for e in evals]),
                "hallucination_rate": statistics.mean([e.hallucination_rate for e in evals])
            }
        
        return breakdown
    
    def _analyze_improvements(self, evaluations: List[TaskEvaluation]) -> Dict[str, Any]:
        """Analyze improvements across prompt versions"""
        version_groups = {}
        
        for eval in evaluations:
            version = eval.prompt_version
            if version not in version_groups:
                version_groups[version] = []
            version_groups[version].append(eval)
        
        analysis = {
            "versions_tested": list(version_groups.keys()),
            "version_performance": {}
        }
        
        for version, evals in version_groups.items():
            analysis["version_performance"][version] = {
                "count": len(evals),
                "accuracy": statistics.mean([e.accuracy_score for e in evals]),
                "coherence": statistics.mean([e.reasoning_coherence for e in evals])
            }
        
        # Calculate improvement trends
        if len(version_groups) > 1:
            versions = sorted(version_groups.keys())
            accuracy_trend = []
            for v in versions:
                accuracy_trend.append(analysis["version_performance"][v]["accuracy"])
            
            analysis["improvement_trend"] = {
                "accuracy_improvement": accuracy_trend[-1] - accuracy_trend[0] if len(accuracy_trend) > 1 else 0.0,
                "consistent_improvement": all(accuracy_trend[i] <= accuracy_trend[i+1] for i in range(len(accuracy_trend)-1))
            }
        
        return analysis
    
    def _generate_recommendations(self, 
                                evaluations: List[TaskEvaluation], 
                                metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Accuracy recommendations
        if metrics["overall_accuracy"] < 0.5:
            recommendations.append("Overall accuracy is low. Consider improving prompt clarity and specificity.")
        
        # Coherence recommendations
        if metrics["overall_coherence"] < 0.6:
            recommendations.append("Reasoning coherence needs improvement. Focus on step-by-step prompting.")
        
        # Hallucination recommendations
        if metrics["overall_hallucination_rate"] > 0.3:
            recommendations.append("High hallucination rate detected. Consider more constrained prompting.")
        
        # Performance recommendations
        if metrics["average_execution_time"] > 30:
            recommendations.append("Execution time is high. Consider reducing reasoning depth or paths.")
        
        # Domain-specific recommendations
        domain_breakdown = self._calculate_domain_breakdown(evaluations)
        for domain, domain_metrics in domain_breakdown.items():
            if domain_metrics["accuracy"] < 0.4:
                recommendations.append(f"Poor performance in {domain} domain. Consider domain-specific prompt optimization.")
        
        # Consistency recommendations
        if metrics["accuracy_std"] > 0.3:
            recommendations.append("High variance in accuracy. Focus on consistent prompt performance.")
        
        if not recommendations:
            recommendations.append("Performance is satisfactory. Continue monitoring and fine-tuning.")
        
        return recommendations
    
    def save_evaluation_report(self, report: EvaluationReport) -> str:
        """Save evaluation report to file"""
        # Save as JSON
        json_filename = f"{report.report_id}.json"
        json_filepath = self.evaluation_dir / json_filename

        # Convert to dictionary for JSON serialization
        report_dict = asdict(report)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        # Also save as Markdown
        md_filename = f"{report.report_id}.md"
        md_filepath = self.evaluation_dir / md_filename

        markdown_content = self._generate_markdown_report(report)
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Evaluation report saved: {json_filepath} and {md_filepath}")
        return str(json_filepath)

    def _generate_markdown_report(self, report: EvaluationReport) -> str:
        """Generate markdown version of the evaluation report"""
        md_lines = []

        # Header
        md_lines.append(f"# Evaluation Report: {report.report_id}")
        md_lines.append(f"**Generated:** {report.evaluation_timestamp}")
        md_lines.append(f"**Total Tasks:** {report.total_tasks}")
        md_lines.append("")

        # Executive Summary
        md_lines.append("## Executive Summary")
        md_lines.append("")
        metrics = report.aggregate_metrics
        md_lines.append(f"- **Overall Accuracy:** {metrics.get('overall_accuracy', 0):.1%}")
        md_lines.append(f"- **Reasoning Coherence:** {metrics.get('overall_coherence', 0):.1%}")
        md_lines.append(f"- **Hallucination Rate:** {metrics.get('overall_hallucination_rate', 0):.1%}")
        md_lines.append(f"- **Average Execution Time:** {metrics.get('average_execution_time', 0):.2f}s")
        md_lines.append(f"- **Average Consensus Score:** {metrics.get('average_consensus_score', 0):.3f}")
        md_lines.append("")

        # Domain Breakdown
        if report.domain_breakdown:
            md_lines.append("## Performance by Domain")
            md_lines.append("")
            md_lines.append("| Domain | Tasks | Accuracy | Coherence | Hallucination | Avg Time |")
            md_lines.append("|--------|-------|----------|-----------|---------------|----------|")

            for domain, domain_metrics in report.domain_breakdown.items():
                md_lines.append(f"| {domain} | {domain_metrics['count']} | {domain_metrics['accuracy']:.1%} | {domain_metrics['coherence']:.1%} | {domain_metrics['hallucination_rate']:.1%} | {domain_metrics['execution_time']:.2f}s |")
            md_lines.append("")

        # Difficulty Breakdown
        if report.difficulty_breakdown:
            md_lines.append("## Performance by Difficulty")
            md_lines.append("")
            md_lines.append("| Difficulty | Tasks | Accuracy | Coherence | Hallucination |")
            md_lines.append("|------------|-------|----------|-----------|---------------|")

            for difficulty, diff_metrics in report.difficulty_breakdown.items():
                md_lines.append(f"| {difficulty} | {diff_metrics['count']} | {diff_metrics['accuracy']:.1%} | {diff_metrics['coherence']:.1%} | {diff_metrics['hallucination_rate']:.1%} |")
            md_lines.append("")

        # Recommendations
        if report.recommendations:
            md_lines.append("## Recommendations")
            md_lines.append("")
            for i, rec in enumerate(report.recommendations, 1):
                md_lines.append(f"{i}. {rec}")
            md_lines.append("")

        # Detailed Results
        md_lines.append("## Detailed Task Results")
        md_lines.append("")
        md_lines.append("| Task ID | Domain | Difficulty | Accuracy | Coherence | Time | Expected | Generated |")
        md_lines.append("|---------|--------|------------|----------|-----------|------|----------|-----------|")

        for eval in report.task_evaluations:
            expected_short = eval.expected_answer[:30] + "..." if len(eval.expected_answer) > 30 else eval.expected_answer
            generated_short = eval.generated_answer[:30] + "..." if len(eval.generated_answer) > 30 else eval.generated_answer

            md_lines.append(f"| {eval.task_id} | {eval.task_domain} | {eval.task_difficulty} | {eval.accuracy_score:.1%} | {eval.reasoning_coherence:.1%} | {eval.execution_time:.1f}s | {expected_short} | {generated_short} |")

        md_lines.append("")

        # Footer
        md_lines.append("---")
        md_lines.append("*Report generated by Prompt Engineering Pipeline Evaluation System*")

        return "\n".join(md_lines)


# Example usage and testing
if __name__ == "__main__":
    # Test the evaluator
    evaluator = Evaluator()
    
    print("Evaluator initialized and ready for use")
    print(f"Evaluation directory: {evaluator.evaluation_dir}")
