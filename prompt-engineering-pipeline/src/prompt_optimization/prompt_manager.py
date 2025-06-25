"""
Prompt Manager
Handles versioning, storage, and management of prompts and their variants
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import asdict

from .optimizer import PromptVariant, OptimizationResult


class PromptManager:
    """Manages prompt versions, storage, and retrieval"""
    
    def __init__(self, prompts_dir: str = "prompts", logs_dir: str = "logs"):
        """
        Initialize PromptManager
        
        Args:
            prompts_dir: Directory for storing prompt files
            logs_dir: Directory for storing optimization logs
        """
        self.prompts_dir = Path(prompts_dir)
        self.logs_dir = Path(logs_dir)
        self.optimization_logs_dir = self.logs_dir / "optimization"
        
        # Create directories if they don't exist
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize prompt registry
        self.registry_file = self.prompts_dir / "prompt_registry.json"
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize the prompt registry file if it doesn't exist"""
        if not self.registry_file.exists():
            initial_registry = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "prompts": {}
            }
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(initial_registry, f, indent=2, ensure_ascii=False)
    
    def load_registry(self) -> Dict[str, Any]:
        """Load the prompt registry"""
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_registry(self, registry: Dict[str, Any]):
        """Save the prompt registry"""
        registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    
    def get_default_prompts(self) -> Dict[str, str]:
        """Get default prompts for different domains"""
        return {
            "math": "Solve this math problem step by step. Show your work and provide the final answer.",
            "logic": "Analyze this logic problem carefully. State your assumptions and reasoning clearly.",
            "code_debugging": "Debug this code by identifying the issue and explaining the fix needed.",
            "word_problems": "Read this word problem carefully and solve it step by step.",
            "pattern_recognition": "Identify the pattern in this sequence and find the next element.",
            "geometry": "Solve this geometry problem by applying relevant formulas and showing calculations.",
            "probability": "Calculate the probability by identifying all possible outcomes and favorable cases.",
            "general": "Analyze this problem systematically and provide a clear solution."
        }
    
    def create_initial_prompt(self, task_data: Dict[str, Any]) -> str:
        """
        Create an initial prompt for a task based on its domain
        
        Args:
            task_data: Task information
            
        Returns:
            Initial prompt string
        """
        domain = task_data.get("domain", "general")
        default_prompts = self.get_default_prompts()
        
        base_prompt = default_prompts.get(domain, default_prompts["general"])
        
        # Customize based on task specifics
        if "title" in task_data:
            base_prompt = f"Task: {task_data['title']}\n\n{base_prompt}"
        
        return base_prompt
    
    def save_prompt_version(self, 
                           task_id: str, 
                           prompt_content: str, 
                           version: int,
                           performance_score: float = 0.0,
                           metadata: Dict[str, Any] = None) -> str:
        """
        Save a prompt version to file
        
        Args:
            task_id: Task identifier
            prompt_content: Prompt content
            version: Version number
            performance_score: Performance score of this prompt
            metadata: Additional metadata
            
        Returns:
            Path to saved prompt file
        """
        # Create filename
        filename = f"{task_id}_v{version}.txt"
        filepath = self.prompts_dir / filename
        
        # Create prompt data
        prompt_data = {
            "task_id": task_id,
            "version": version,
            "content": prompt_content,
            "performance_score": performance_score,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save prompt content to text file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        
        # Save metadata to JSON file
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
        
        # Update registry
        registry = self.load_registry()
        
        if task_id not in registry["prompts"]:
            registry["prompts"][task_id] = {
                "versions": [],
                "best_version": version,
                "best_score": performance_score
            }
        
        # Add version info
        version_info = {
            "version": version,
            "filename": filename,
            "performance_score": performance_score,
            "created_at": prompt_data["created_at"]
        }
        
        registry["prompts"][task_id]["versions"].append(version_info)
        
        # Update best version if this one is better
        if performance_score > registry["prompts"][task_id]["best_score"]:
            registry["prompts"][task_id]["best_version"] = version
            registry["prompts"][task_id]["best_score"] = performance_score
        
        self.save_registry(registry)
        
        print(f"Prompt version saved: {filepath}")
        return str(filepath)
    
    def load_prompt_version(self, task_id: str, version: int = None) -> Optional[str]:
        """
        Load a specific prompt version
        
        Args:
            task_id: Task identifier
            version: Version number (if None, loads best version)
            
        Returns:
            Prompt content or None if not found
        """
        registry = self.load_registry()
        
        if task_id not in registry["prompts"]:
            return None
        
        # Determine which version to load
        if version is None:
            version = registry["prompts"][task_id]["best_version"]
        
        # Find the version file
        filename = f"{task_id}_v{version}.txt"
        filepath = self.prompts_dir / filename
        
        if not filepath.exists():
            return None
        
        # Load prompt content
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_prompt_history(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get the version history for a task's prompts
        
        Args:
            task_id: Task identifier
            
        Returns:
            List of version information
        """
        registry = self.load_registry()
        
        if task_id not in registry["prompts"]:
            return []
        
        return registry["prompts"][task_id]["versions"]
    
    def save_optimization_result(self, result: OptimizationResult) -> str:
        """
        Save optimization result to log file
        
        Args:
            result: OptimizationResult to save
            
        Returns:
            Path to saved log file
        """
        # Convert result to dictionary
        result_dict = {
            "task_id": result.task_id,
            "original_prompt": result.original_prompt,
            "best_prompt": result.best_prompt,
            "improvement_score": result.improvement_score,
            "optimization_iterations": result.optimization_iterations,
            "execution_time": result.execution_time,
            "generated_variants": [
                {
                    "id": variant.id,
                    "content": variant.content,
                    "version": variant.version,
                    "parent_id": variant.parent_id,
                    "performance_score": variant.performance_score,
                    "generation_method": variant.generation_method,
                    "timestamp": variant.timestamp,
                    "metadata": variant.metadata
                }
                for variant in result.generated_variants
            ],
            "saved_at": datetime.now().isoformat()
        }
        
        # Create filename
        filename = f"{result.task_id}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.optimization_logs_dir / filename
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Save the best prompt as a new version
        if result.improvement_score > 0:
            # Find the next version number
            registry = self.load_registry()
            if result.task_id in registry["prompts"]:
                next_version = len(registry["prompts"][result.task_id]["versions"]) + 1
            else:
                next_version = 1
            
            # Save the optimized prompt
            self.save_prompt_version(
                task_id=result.task_id,
                prompt_content=result.best_prompt,
                version=next_version,
                performance_score=result.improvement_score,
                metadata={
                    "optimization_method": "automated",
                    "original_score": 0.0,  # Would need to track this
                    "iterations": result.optimization_iterations,
                    "optimization_log": str(filepath)
                }
            )
        
        print(f"Optimization result saved: {filepath}")
        return str(filepath)
    
    def list_tasks_with_prompts(self) -> List[str]:
        """Get list of task IDs that have prompts"""
        registry = self.load_registry()
        return list(registry["prompts"].keys())
    
    def get_best_prompts_summary(self) -> Dict[str, Any]:
        """Get summary of best prompts for all tasks"""
        registry = self.load_registry()
        summary = {}
        
        for task_id, task_info in registry["prompts"].items():
            summary[task_id] = {
                "best_version": task_info["best_version"],
                "best_score": task_info["best_score"],
                "total_versions": len(task_info["versions"]),
                "latest_update": task_info["versions"][-1]["created_at"] if task_info["versions"] else None
            }
        
        return summary
    
    def cleanup_old_versions(self, task_id: str, keep_versions: int = 5):
        """
        Clean up old prompt versions, keeping only the most recent ones
        
        Args:
            task_id: Task identifier
            keep_versions: Number of versions to keep
        """
        registry = self.load_registry()
        
        if task_id not in registry["prompts"]:
            return
        
        versions = registry["prompts"][task_id]["versions"]
        
        if len(versions) <= keep_versions:
            return
        
        # Sort by version number and keep the most recent
        versions.sort(key=lambda x: x["version"])
        versions_to_remove = versions[:-keep_versions]
        
        # Remove old files
        for version_info in versions_to_remove:
            filename = version_info["filename"]
            filepath = self.prompts_dir / filename
            metadata_file = filepath.with_suffix('.json')
            
            # Remove files if they exist
            if filepath.exists():
                filepath.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
        
        # Update registry
        registry["prompts"][task_id]["versions"] = versions[-keep_versions:]
        self.save_registry(registry)
        
        print(f"Cleaned up {len(versions_to_remove)} old versions for task {task_id}")


# Example usage and testing
if __name__ == "__main__":
    # Test the prompt manager
    manager = PromptManager()
    
    # Test task
    test_task = {
        "id": "test_001",
        "domain": "math",
        "title": "Simple Addition"
    }
    
    # Create initial prompt
    initial_prompt = manager.create_initial_prompt(test_task)
    print(f"Initial prompt: {initial_prompt}")
    
    # Save a prompt version
    manager.save_prompt_version(
        task_id=test_task["id"],
        prompt_content=initial_prompt,
        version=1,
        performance_score=0.5
    )
    
    # Load the prompt back
    loaded_prompt = manager.load_prompt_version(test_task["id"], 1)
    print(f"Loaded prompt: {loaded_prompt}")
    
    print("PromptManager test completed successfully")
