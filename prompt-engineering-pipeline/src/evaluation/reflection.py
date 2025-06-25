"""
Reflection System
Analyzes pipeline performance and generates insights for improvement
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
import logging

from .evaluator import EvaluationReport, TaskEvaluation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReflectionInsight:
    """Single insight from reflection analysis"""
    category: str
    insight_type: str
    description: str
    evidence: List[str]
    confidence: float
    actionable_steps: List[str]


@dataclass
class ReflectionReport:
    """Comprehensive reflection report"""
    report_id: str
    reflection_timestamp: str
    evaluation_reports_analyzed: List[str]
    key_insights: List[ReflectionInsight]
    performance_trends: Dict[str, Any]
    success_patterns: Dict[str, Any]
    failure_patterns: Dict[str, Any]
    optimization_opportunities: List[str]
    next_steps: List[str]


class ReflectionSystem:
    """System for analyzing pipeline performance and generating insights"""
    
    def __init__(self, evaluation_dir: str = "evaluation"):
        """
        Initialize Reflection System
        
        Args:
            evaluation_dir: Directory containing evaluation reports
        """
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Reflection System initialized successfully")
    
    def load_evaluation_reports(self, report_ids: List[str] = None) -> List[EvaluationReport]:
        """
        Load evaluation reports for analysis
        
        Args:
            report_ids: Specific report IDs to load (if None, loads all)
            
        Returns:
            List of EvaluationReport objects
        """
        reports = []
        
        if report_ids is None:
            # Load all JSON reports
            json_files = list(self.evaluation_dir.glob("report_*.json"))
        else:
            json_files = [self.evaluation_dir / f"{report_id}.json" for report_id in report_ids]
        
        for json_file in json_files:
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    # Convert back to EvaluationReport object (simplified)
                    # In a full implementation, you'd properly reconstruct the objects
                    reports.append(report_data)
                    
                except Exception as e:
                    logger.error(f"Error loading report {json_file}: {e}")
        
        logger.info(f"Loaded {len(reports)} evaluation reports")
        return reports
    
    def analyze_performance_trends(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance trends across multiple evaluation reports
        
        Args:
            reports: List of evaluation report dictionaries
            
        Returns:
            Performance trends analysis
        """
        if not reports:
            return {}
        
        # Sort reports by timestamp
        sorted_reports = sorted(reports, key=lambda x: x.get('evaluation_timestamp', ''))
        
        trends = {
            "accuracy_trend": [],
            "coherence_trend": [],
            "hallucination_trend": [],
            "execution_time_trend": [],
            "improvement_over_time": {},
            "domain_trends": {},
            "difficulty_trends": {}
        }
        
        # Extract metrics over time
        for report in sorted_reports:
            metrics = report.get('aggregate_metrics', {})
            trends["accuracy_trend"].append(metrics.get('overall_accuracy', 0))
            trends["coherence_trend"].append(metrics.get('overall_coherence', 0))
            trends["hallucination_trend"].append(metrics.get('overall_hallucination_rate', 0))
            trends["execution_time_trend"].append(metrics.get('average_execution_time', 0))
        
        # Calculate improvement rates
        if len(trends["accuracy_trend"]) > 1:
            trends["improvement_over_time"] = {
                "accuracy_improvement": trends["accuracy_trend"][-1] - trends["accuracy_trend"][0],
                "coherence_improvement": trends["coherence_trend"][-1] - trends["coherence_trend"][0],
                "hallucination_reduction": trends["hallucination_trend"][0] - trends["hallucination_trend"][-1],
                "time_efficiency_change": trends["execution_time_trend"][0] - trends["execution_time_trend"][-1]
            }
        
        # Analyze domain trends
        domain_performance = {}
        for report in sorted_reports:
            for domain, domain_metrics in report.get('domain_breakdown', {}).items():
                if domain not in domain_performance:
                    domain_performance[domain] = []
                domain_performance[domain].append(domain_metrics.get('accuracy', 0))
        
        for domain, accuracies in domain_performance.items():
            if len(accuracies) > 1:
                trends["domain_trends"][domain] = {
                    "improvement": accuracies[-1] - accuracies[0],
                    "consistency": statistics.stdev(accuracies) if len(accuracies) > 1 else 0
                }
        
        return trends
    
    def identify_success_patterns(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patterns in successful task completions
        
        Args:
            reports: List of evaluation report dictionaries
            
        Returns:
            Success patterns analysis
        """
        success_patterns = {
            "high_accuracy_tasks": [],
            "successful_domains": [],
            "effective_prompt_versions": [],
            "optimal_reasoning_depth": None,
            "success_characteristics": {}
        }
        
        all_evaluations = []
        for report in reports:
            all_evaluations.extend(report.get('task_evaluations', []))
        
        if not all_evaluations:
            return success_patterns
        
        # Identify high-accuracy tasks
        high_accuracy_threshold = 0.8
        high_accuracy_tasks = [eval for eval in all_evaluations if eval.get('accuracy_score', 0) >= high_accuracy_threshold]
        
        success_patterns["high_accuracy_tasks"] = [
            {
                "task_id": eval.get('task_id'),
                "domain": eval.get('task_domain'),
                "accuracy": eval.get('accuracy_score'),
                "coherence": eval.get('reasoning_coherence')
            }
            for eval in high_accuracy_tasks
        ]
        
        # Identify successful domains
        domain_success = {}
        for eval in all_evaluations:
            domain = eval.get('task_domain', 'unknown')
            if domain not in domain_success:
                domain_success[domain] = []
            domain_success[domain].append(eval.get('accuracy_score', 0))
        
        for domain, accuracies in domain_success.items():
            avg_accuracy = statistics.mean(accuracies)
            if avg_accuracy >= 0.6:  # Threshold for "successful"
                success_patterns["successful_domains"].append({
                    "domain": domain,
                    "average_accuracy": avg_accuracy,
                    "task_count": len(accuracies)
                })
        
        # Analyze prompt version effectiveness
        version_performance = {}
        for eval in all_evaluations:
            version = eval.get('prompt_version', 1)
            if version not in version_performance:
                version_performance[version] = []
            version_performance[version].append(eval.get('accuracy_score', 0))
        
        for version, accuracies in version_performance.items():
            avg_accuracy = statistics.mean(accuracies)
            success_patterns["effective_prompt_versions"].append({
                "version": version,
                "average_accuracy": avg_accuracy,
                "task_count": len(accuracies)
            })
        
        # Find optimal reasoning depth
        depth_performance = {}
        for eval in all_evaluations:
            depth = eval.get('detailed_metrics', {}).get('reasoning_depth', 0)
            if depth not in depth_performance:
                depth_performance[depth] = []
            depth_performance[depth].append(eval.get('accuracy_score', 0))
        
        if depth_performance:
            best_depth = max(depth_performance.keys(), key=lambda d: statistics.mean(depth_performance[d]))
            success_patterns["optimal_reasoning_depth"] = {
                "depth": best_depth,
                "average_accuracy": statistics.mean(depth_performance[best_depth])
            }
        
        return success_patterns
    
    def identify_failure_patterns(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patterns in failed or low-performing tasks
        
        Args:
            reports: List of evaluation report dictionaries
            
        Returns:
            Failure patterns analysis
        """
        failure_patterns = {
            "low_accuracy_tasks": [],
            "problematic_domains": [],
            "common_failure_modes": [],
            "hallucination_hotspots": [],
            "performance_bottlenecks": {}
        }
        
        all_evaluations = []
        for report in reports:
            all_evaluations.extend(report.get('task_evaluations', []))
        
        if not all_evaluations:
            return failure_patterns
        
        # Identify low-accuracy tasks
        low_accuracy_threshold = 0.3
        low_accuracy_tasks = [eval for eval in all_evaluations if eval.get('accuracy_score', 0) <= low_accuracy_threshold]
        
        failure_patterns["low_accuracy_tasks"] = [
            {
                "task_id": eval.get('task_id'),
                "domain": eval.get('task_domain'),
                "accuracy": eval.get('accuracy_score'),
                "hallucination_rate": eval.get('hallucination_rate'),
                "expected": eval.get('expected_answer', '')[:50],
                "generated": eval.get('generated_answer', '')[:50]
            }
            for eval in low_accuracy_tasks
        ]
        
        # Identify problematic domains
        domain_performance = {}
        for eval in all_evaluations:
            domain = eval.get('task_domain', 'unknown')
            if domain not in domain_performance:
                domain_performance[domain] = []
            domain_performance[domain].append(eval.get('accuracy_score', 0))
        
        for domain, accuracies in domain_performance.items():
            avg_accuracy = statistics.mean(accuracies)
            if avg_accuracy <= 0.4:  # Threshold for "problematic"
                failure_patterns["problematic_domains"].append({
                    "domain": domain,
                    "average_accuracy": avg_accuracy,
                    "task_count": len(accuracies)
                })
        
        # Identify hallucination hotspots
        high_hallucination_threshold = 0.5
        high_hallucination_tasks = [eval for eval in all_evaluations if eval.get('hallucination_rate', 0) >= high_hallucination_threshold]
        
        failure_patterns["hallucination_hotspots"] = [
            {
                "task_id": eval.get('task_id'),
                "domain": eval.get('task_domain'),
                "hallucination_rate": eval.get('hallucination_rate'),
                "coherence": eval.get('reasoning_coherence')
            }
            for eval in high_hallucination_tasks
        ]
        
        # Identify performance bottlenecks
        execution_times = [eval.get('execution_time', 0) for eval in all_evaluations]
        if execution_times:
            avg_time = statistics.mean(execution_times)
            slow_tasks = [eval for eval in all_evaluations if eval.get('execution_time', 0) > avg_time * 1.5]
            
            failure_patterns["performance_bottlenecks"] = {
                "average_execution_time": avg_time,
                "slow_tasks": [
                    {
                        "task_id": eval.get('task_id'),
                        "execution_time": eval.get('execution_time'),
                        "reasoning_paths": eval.get('detailed_metrics', {}).get('num_reasoning_paths', 0)
                    }
                    for eval in slow_tasks
                ]
            }
        
        return failure_patterns
    
    def generate_insights(self, 
                         trends: Dict[str, Any],
                         success_patterns: Dict[str, Any],
                         failure_patterns: Dict[str, Any]) -> List[ReflectionInsight]:
        """
        Generate actionable insights from analysis
        
        Args:
            trends: Performance trends analysis
            success_patterns: Success patterns analysis
            failure_patterns: Failure patterns analysis
            
        Returns:
            List of ReflectionInsight objects
        """
        insights = []
        
        # Trend-based insights
        if trends.get("improvement_over_time", {}).get("accuracy_improvement", 0) > 0.1:
            insights.append(ReflectionInsight(
                category="Performance Trends",
                insight_type="Positive Trend",
                description="Accuracy is improving over time through optimization",
                evidence=[f"Accuracy improved by {trends['improvement_over_time']['accuracy_improvement']:.1%}"],
                confidence=0.8,
                actionable_steps=["Continue current optimization approach", "Monitor for plateau effects"]
            ))
        
        # Success pattern insights
        successful_domains = success_patterns.get("successful_domains", [])
        if successful_domains:
            best_domain = max(successful_domains, key=lambda x: x["average_accuracy"])
            insights.append(ReflectionInsight(
                category="Success Patterns",
                insight_type="Domain Excellence",
                description=f"Strong performance in {best_domain['domain']} domain",
                evidence=[f"Average accuracy: {best_domain['average_accuracy']:.1%}"],
                confidence=0.9,
                actionable_steps=[
                    f"Analyze successful {best_domain['domain']} prompts",
                    "Apply successful patterns to other domains"
                ]
            ))
        
        # Failure pattern insights
        problematic_domains = failure_patterns.get("problematic_domains", [])
        if problematic_domains:
            worst_domain = min(problematic_domains, key=lambda x: x["average_accuracy"])
            insights.append(ReflectionInsight(
                category="Failure Patterns",
                insight_type="Domain Weakness",
                description=f"Poor performance in {worst_domain['domain']} domain",
                evidence=[f"Average accuracy: {worst_domain['average_accuracy']:.1%}"],
                confidence=0.9,
                actionable_steps=[
                    f"Focus optimization efforts on {worst_domain['domain']} tasks",
                    "Consider domain-specific prompting strategies",
                    "Increase reasoning depth for complex domains"
                ]
            ))
        
        # Hallucination insights
        hallucination_hotspots = failure_patterns.get("hallucination_hotspots", [])
        if len(hallucination_hotspots) > 0:
            insights.append(ReflectionInsight(
                category="Quality Issues",
                insight_type="Hallucination Problem",
                description="High hallucination rate detected in multiple tasks",
                evidence=[f"{len(hallucination_hotspots)} tasks with high hallucination"],
                confidence=0.8,
                actionable_steps=[
                    "Implement stricter prompt constraints",
                    "Add verification steps to reasoning",
                    "Consider temperature reduction in generation"
                ]
            ))
        
        return insights
    
    def generate_reflection_report(self, report_ids: List[str] = None) -> ReflectionReport:
        """
        Generate comprehensive reflection report
        
        Args:
            report_ids: Specific evaluation report IDs to analyze
            
        Returns:
            ReflectionReport object
        """
        logger.info("Generating reflection report")
        
        # Load evaluation reports
        reports = self.load_evaluation_reports(report_ids)
        
        if not reports:
            logger.warning("No evaluation reports found for reflection")
            return ReflectionReport(
                report_id="empty_reflection",
                reflection_timestamp=datetime.now().isoformat(),
                evaluation_reports_analyzed=[],
                key_insights=[],
                performance_trends={},
                success_patterns={},
                failure_patterns={},
                optimization_opportunities=[],
                next_steps=["Generate evaluation reports first"]
            )
        
        # Perform analysis
        trends = self.analyze_performance_trends(reports)
        success_patterns = self.identify_success_patterns(reports)
        failure_patterns = self.identify_failure_patterns(reports)
        
        # Generate insights
        insights = self.generate_insights(trends, success_patterns, failure_patterns)
        
        # Generate optimization opportunities
        optimization_opportunities = self._generate_optimization_opportunities(
            trends, success_patterns, failure_patterns
        )
        
        # Generate next steps
        next_steps = self._generate_next_steps(insights, optimization_opportunities)
        
        # Create reflection report
        report = ReflectionReport(
            report_id=f"reflection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reflection_timestamp=datetime.now().isoformat(),
            evaluation_reports_analyzed=[r.get('report_id', 'unknown') for r in reports],
            key_insights=insights,
            performance_trends=trends,
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            optimization_opportunities=optimization_opportunities,
            next_steps=next_steps
        )
        
        logger.info("Reflection report generated successfully")
        return report
    
    def _generate_optimization_opportunities(self, 
                                           trends: Dict[str, Any],
                                           success_patterns: Dict[str, Any],
                                           failure_patterns: Dict[str, Any]) -> List[str]:
        """Generate optimization opportunities"""
        opportunities = []
        
        # Based on failure patterns
        if failure_patterns.get("problematic_domains"):
            opportunities.append("Focus optimization on underperforming domains")
        
        if failure_patterns.get("hallucination_hotspots"):
            opportunities.append("Implement hallucination reduction techniques")
        
        # Based on success patterns
        if success_patterns.get("successful_domains"):
            opportunities.append("Transfer successful domain strategies to other areas")
        
        # Based on trends
        if trends.get("improvement_over_time", {}).get("accuracy_improvement", 0) < 0:
            opportunities.append("Investigate causes of performance degradation")
        
        return opportunities
    
    def _generate_next_steps(self, 
                           insights: List[ReflectionInsight],
                           opportunities: List[str]) -> List[str]:
        """Generate actionable next steps"""
        next_steps = []
        
        # Collect actionable steps from insights
        for insight in insights:
            next_steps.extend(insight.actionable_steps)
        
        # Add opportunity-based steps
        next_steps.extend(opportunities)
        
        # Remove duplicates and prioritize
        unique_steps = list(set(next_steps))
        
        return unique_steps[:10]  # Limit to top 10 steps
    
    def save_reflection_report(self, report: ReflectionReport) -> str:
        """Save reflection report to file"""
        # Save as JSON
        json_filename = f"{report.report_id}.json"
        json_filepath = self.evaluation_dir / json_filename
        
        # Convert to dictionary for JSON serialization
        report_dict = asdict(report)
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        # Save as Markdown
        md_filename = "reflection.md"
        md_filepath = self.evaluation_dir / md_filename
        
        markdown_content = self._generate_reflection_markdown(report)
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Reflection report saved: {json_filepath} and {md_filepath}")
        return str(json_filepath)
    
    def _generate_reflection_markdown(self, report: ReflectionReport) -> str:
        """Generate markdown version of reflection report"""
        md_lines = []
        
        # Header
        md_lines.append("# Pipeline Reflection Report")
        md_lines.append(f"**Generated:** {report.reflection_timestamp}")
        md_lines.append(f"**Reports Analyzed:** {len(report.evaluation_reports_analyzed)}")
        md_lines.append("")
        
        # Key Insights
        md_lines.append("## Key Insights")
        md_lines.append("")
        for insight in report.key_insights:
            md_lines.append(f"### {insight.category}: {insight.insight_type}")
            md_lines.append(f"**Description:** {insight.description}")
            md_lines.append(f"**Confidence:** {insight.confidence:.1%}")
            md_lines.append("**Evidence:**")
            for evidence in insight.evidence:
                md_lines.append(f"- {evidence}")
            md_lines.append("**Actionable Steps:**")
            for step in insight.actionable_steps:
                md_lines.append(f"- {step}")
            md_lines.append("")
        
        # Optimization Opportunities
        md_lines.append("## Optimization Opportunities")
        md_lines.append("")
        for i, opportunity in enumerate(report.optimization_opportunities, 1):
            md_lines.append(f"{i}. {opportunity}")
        md_lines.append("")
        
        # Next Steps
        md_lines.append("## Recommended Next Steps")
        md_lines.append("")
        for i, step in enumerate(report.next_steps, 1):
            md_lines.append(f"{i}. {step}")
        md_lines.append("")
        
        # Footer
        md_lines.append("---")
        md_lines.append("*Generated by Prompt Engineering Pipeline Reflection System*")
        
        return "\n".join(md_lines)


# Example usage and testing
if __name__ == "__main__":
    # Test the reflection system
    reflection_system = ReflectionSystem()
    
    print("Reflection System initialized and ready for use")
    print(f"Evaluation directory: {reflection_system.evaluation_dir}")
