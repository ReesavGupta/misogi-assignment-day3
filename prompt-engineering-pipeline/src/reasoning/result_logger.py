"""
Result Logger for Tree-of-Thought Reasoning
Handles saving and loading of reasoning results
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import asdict

from .tot_engine import ToTResult, ReasoningPath, ThoughtNode


class ResultLogger:
    """Handles logging and persistence of ToT reasoning results"""
    
    def __init__(self, logs_dir: str = "logs"):
        """
        Initialize ResultLogger
        
        Args:
            logs_dir: Directory for storing logs
        """
        self.logs_dir = Path(logs_dir)
        self.reasoning_paths_dir = self.logs_dir / "reasoning_paths"
        
        # Create directories if they don't exist
        self.reasoning_paths_dir.mkdir(parents=True, exist_ok=True)
        
    def save_result(self, result: ToTResult) -> str:
        """
        Save ToT result to JSON file
        
        Args:
            result: ToTResult to save
            
        Returns:
            Path to saved file
        """
        # Convert result to dictionary
        result_dict = self._result_to_dict(result)
        
        # Add metadata
        result_dict["saved_at"] = datetime.now().isoformat()
        result_dict["version"] = "1.0"
        
        # Create filename
        filename = f"{result.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reasoning_paths_dir / filename
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
        print(f"Result saved to: {filepath}")
        return str(filepath)
    
    def _result_to_dict(self, result: ToTResult) -> Dict[str, Any]:
        """Convert ToTResult to dictionary for JSON serialization"""
        return {
            "task_id": result.task_id,
            "final_answer": result.final_answer,
            "consensus_score": result.consensus_score,
            "execution_time": result.execution_time,
            "model_info": result.model_info,
            "reasoning_paths": [
                {
                    "path_id": path.path_id,
                    "final_answer": path.final_answer,
                    "confidence_score": path.confidence_score,
                    "reasoning_quality": path.reasoning_quality,
                    "nodes": [
                        {
                            "id": node.id,
                            "content": node.content,
                            "parent_id": node.parent_id,
                            "depth": node.depth,
                            "confidence": node.confidence,
                            "timestamp": node.timestamp
                        }
                        for node in path.nodes
                    ]
                }
                for path in result.reasoning_paths
            ]
        }
    
    def load_result(self, filepath: str) -> Dict[str, Any]:
        """
        Load ToT result from JSON file
        
        Args:
            filepath: Path to result file
            
        Returns:
            Result dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_results(self, task_id: str = None) -> List[str]:
        """
        List all result files, optionally filtered by task_id
        
        Args:
            task_id: Optional task ID to filter by
            
        Returns:
            List of result file paths
        """
        pattern = f"{task_id}_*.json" if task_id else "*.json"
        return [str(f) for f in self.reasoning_paths_dir.glob(pattern)]
    
    def create_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary report from multiple results
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Summary report dictionary
        """
        if not results:
            return {"error": "No results provided"}
            
        summary = {
            "report_generated_at": datetime.now().isoformat(),
            "total_tasks": len(results),
            "average_execution_time": sum(r["execution_time"] for r in results) / len(results),
            "average_consensus_score": sum(r["consensus_score"] for r in results) / len(results),
            "tasks_summary": []
        }
        
        for result in results:
            task_summary = {
                "task_id": result["task_id"],
                "final_answer": result["final_answer"],
                "consensus_score": result["consensus_score"],
                "execution_time": result["execution_time"],
                "num_reasoning_paths": len(result["reasoning_paths"]),
                "avg_path_quality": sum(
                    path["reasoning_quality"] for path in result["reasoning_paths"]
                ) / len(result["reasoning_paths"]) if result["reasoning_paths"] else 0
            }
            summary["tasks_summary"].append(task_summary)
        
        return summary
    
    def save_summary_report(self, summary: Dict[str, Any], filename: str = None) -> str:
        """
        Save summary report to file
        
        Args:
            summary: Summary report dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"Summary report saved to: {filepath}")
        return str(filepath)
    
    def export_reasoning_trace(self, result: ToTResult, format: str = "markdown") -> str:
        """
        Export reasoning trace in human-readable format
        
        Args:
            result: ToTResult to export
            format: Export format ("markdown" or "text")
            
        Returns:
            Formatted reasoning trace
        """
        if format == "markdown":
            return self._export_markdown(result)
        else:
            return self._export_text(result)
    
    def _export_markdown(self, result: ToTResult) -> str:
        """Export reasoning trace as Markdown"""
        md_content = []
        
        # Header
        md_content.append(f"# Reasoning Trace: {result.task_id}")
        md_content.append(f"**Final Answer:** {result.final_answer}")
        md_content.append(f"**Consensus Score:** {result.consensus_score:.3f}")
        md_content.append(f"**Execution Time:** {result.execution_time:.2f}s")
        md_content.append("")
        
        # Reasoning paths
        for i, path in enumerate(result.reasoning_paths, 1):
            md_content.append(f"## Path {i}: {path.path_id}")
            md_content.append(f"**Final Answer:** {path.final_answer}")
            md_content.append(f"**Confidence:** {path.confidence_score:.3f}")
            md_content.append(f"**Quality:** {path.reasoning_quality:.3f}")
            md_content.append("")
            
            # Nodes
            for node in path.nodes:
                md_content.append(f"### Step {node.depth + 1}")
                md_content.append(f"**Content:** {node.content}")
                md_content.append(f"**Confidence:** {node.confidence:.3f}")
                md_content.append("")
        
        return "\n".join(md_content)
    
    def _export_text(self, result: ToTResult) -> str:
        """Export reasoning trace as plain text"""
        text_content = []
        
        # Header
        text_content.append(f"REASONING TRACE: {result.task_id}")
        text_content.append("=" * 50)
        text_content.append(f"Final Answer: {result.final_answer}")
        text_content.append(f"Consensus Score: {result.consensus_score:.3f}")
        text_content.append(f"Execution Time: {result.execution_time:.2f}s")
        text_content.append("")
        
        # Reasoning paths
        for i, path in enumerate(result.reasoning_paths, 1):
            text_content.append(f"PATH {i}: {path.path_id}")
            text_content.append("-" * 30)
            text_content.append(f"Final Answer: {path.final_answer}")
            text_content.append(f"Confidence: {path.confidence_score:.3f}")
            text_content.append(f"Quality: {path.reasoning_quality:.3f}")
            text_content.append("")
            
            # Nodes
            for node in path.nodes:
                text_content.append(f"  Step {node.depth + 1}:")
                text_content.append(f"    {node.content}")
                text_content.append(f"    Confidence: {node.confidence:.3f}")
                text_content.append("")
        
        return "\n".join(text_content)


# Example usage
if __name__ == "__main__":
    # This would typically be used with actual ToT results
    print("ResultLogger utility ready for use with ToT results")
    
    # Test directory creation
    logger = ResultLogger()
    print(f"Logs directory: {logger.logs_dir}")
    print(f"Reasoning paths directory: {logger.reasoning_paths_dir}")
