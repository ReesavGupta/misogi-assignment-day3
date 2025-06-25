"""
Task Loader Module for Prompt Engineering Pipeline
Handles loading and validation of reasoning tasks
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path


class TaskLoader:
    """Loads and manages reasoning tasks from the tasks directory"""
    
    def __init__(self, tasks_dir: str = "tasks"):
        """
        Initialize TaskLoader
        
        Args:
            tasks_dir: Directory containing task files
        """
        self.tasks_dir = Path(tasks_dir)
        self.registry_file = self.tasks_dir / "task_registry.json"
        self.tasks_cache = {}
        
    def load_registry(self) -> Dict[str, Any]:
        """Load the task registry file"""
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Task registry not found at {self.registry_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in task registry: {e}")
    
    def load_task(self, task_id: str) -> Dict[str, Any]:
        """
        Load a specific task by ID
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Task data dictionary
        """
        if task_id in self.tasks_cache:
            return self.tasks_cache[task_id]
            
        registry = self.load_registry()
        
        # Find task in registry
        task_info = None
        for task in registry["task_registry"]["tasks"]:
            if task["id"] == task_id:
                task_info = task
                break
                
        if not task_info:
            raise ValueError(f"Task {task_id} not found in registry")
            
        # Load task file
        task_file = self.tasks_dir / task_info["file"]
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
                
            # Cache the task
            self.tasks_cache[task_id] = task_data
            return task_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Task file not found: {task_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in task file {task_file}: {e}")
    
    def load_all_tasks(self) -> List[Dict[str, Any]]:
        """Load all tasks from the registry"""
        registry = self.load_registry()
        tasks = []
        
        for task_info in registry["task_registry"]["tasks"]:
            task_data = self.load_task(task_info["id"])
            tasks.append(task_data)
            
        return tasks
    
    def get_tasks_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get all tasks from a specific domain"""
        all_tasks = self.load_all_tasks()
        return [task for task in all_tasks if task["domain"] == domain]
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get all tasks of a specific difficulty level"""
        all_tasks = self.load_all_tasks()
        return [task for task in all_tasks if task["difficulty"] == difficulty]
    
    def validate_task(self, task_data: Dict[str, Any]) -> bool:
        """
        Validate that a task has all required fields
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            "id", "domain", "title", "problem", 
            "expected_solution", "expected_reasoning",
            "evaluation_rule", "difficulty", "tags"
        ]
        
        for field in required_fields:
            if field not in task_data:
                print(f"Missing required field: {field}")
                return False
                
        return True
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary statistics about all tasks"""
        registry = self.load_registry()
        return {
            "total_tasks": registry["task_registry"]["total_tasks"],
            "domains": registry["task_registry"]["domains"],
            "difficulty_distribution": registry["task_registry"]["difficulty_distribution"],
            "evaluation_rules": list(registry["task_registry"]["evaluation_rules"].keys())
        }


# Example usage and testing
if __name__ == "__main__":
    # Use relative path from project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tasks_dir = os.path.join(project_root, "tasks")
    loader = TaskLoader(tasks_dir)
    
    # Test loading registry
    try:
        summary = loader.get_task_summary()
        print("Task Summary:")
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Domains: {summary['domains']}")
        print(f"Difficulty distribution: {summary['difficulty_distribution']}")
        
        # Test loading a specific task
        task = loader.load_task("math_001")
        print(f"\nLoaded task: {task['title']}")
        print(f"Domain: {task['domain']}")
        print(f"Problem: {task['problem'][:100]}...")
        
        # Validate task
        is_valid = loader.validate_task(task)
        print(f"Task validation: {'PASS' if is_valid else 'FAIL'}")
        
    except Exception as e:
        print(f"Error: {e}")
