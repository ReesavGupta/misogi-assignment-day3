"""
Automated Prompt Optimizer
Implements OPRO/TextGrad-style feedback loops for prompt improvement
"""

import json
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """Represents a prompt variant with performance metrics"""
    id: str
    content: str
    version: int
    parent_id: Optional[str]
    performance_score: float
    task_id: str
    generation_method: str
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Result from prompt optimization process"""
    task_id: str
    original_prompt: str
    best_prompt: str
    improvement_score: float
    optimization_iterations: int
    generated_variants: List[PromptVariant]
    execution_time: float


class PromptOptimizer:
    """Automated prompt optimization using meta-prompting and feedback loops"""
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 max_iterations: int = 5,
                 min_improvement_threshold: float = 0.1):
        """
        Initialize Prompt Optimizer
        
        Args:
            model_name: HuggingFace model name for meta-prompting
            max_iterations: Maximum optimization iterations
            min_improvement_threshold: Minimum improvement to continue optimization
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.min_improvement_threshold = min_improvement_threshold
        
        # Initialize model and tokenizer for meta-prompting
        logger.info(f"Loading meta-prompting model: {model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"Meta-prompting model loaded on device: {self.device}")
        
    def create_meta_prompts(self, 
                           task_data: Dict[str, Any], 
                           current_prompt: str,
                           performance_feedback: str) -> List[str]:
        """
        Create meta-prompts for generating improved prompt variants
        
        Args:
            task_data: Task information
            current_prompt: Current prompt that needs improvement
            performance_feedback: Feedback on current prompt performance
            
        Returns:
            List of meta-prompts for generating variants
        """
        domain = task_data.get("domain", "general")
        problem = task_data.get("problem", "")
        
        meta_prompts = [
            # Clarity improvement meta-prompt
            f"""The following prompt for a {domain} problem is unclear and needs improvement:

Original prompt: "{current_prompt}"
Problem: {problem}
Performance feedback: {performance_feedback}

Rewrite this prompt to be clearer and more specific. Focus on:
1. Clear step-by-step instructions
2. Explicit reasoning requirements
3. Domain-specific terminology

Improved prompt:""",

            # Structure improvement meta-prompt
            f"""This prompt for solving {domain} problems lacks good structure:

Current prompt: "{current_prompt}"
Task: {problem}
Issues: {performance_feedback}

Create a better-structured prompt that:
1. Breaks down the problem systematically
2. Guides the reasoning process
3. Asks for explicit calculations or logic steps

Better prompt:""",

            # Domain-specific improvement meta-prompt
            f"""The following prompt doesn't effectively leverage {domain} problem-solving techniques:

Prompt: "{current_prompt}"
Problem type: {domain}
Performance issues: {performance_feedback}

Rewrite this prompt to better utilize {domain}-specific approaches:
1. Use appropriate {domain} terminology
2. Reference relevant {domain} methods
3. Structure reasoning according to {domain} best practices

Enhanced prompt:""",

            # Reasoning-focused improvement meta-prompt
            f"""This prompt doesn't elicit good reasoning for the following problem:

Current prompt: "{current_prompt}"
Problem: {problem}
Reasoning issues: {performance_feedback}

Create a prompt that better encourages step-by-step reasoning:
1. Ask for explicit reasoning steps
2. Request verification of each step
3. Encourage showing work and calculations

Reasoning-focused prompt:""",

            # Precision improvement meta-prompt
            f"""The following prompt produces imprecise answers:

Prompt: "{current_prompt}"
Problem: {problem}
Precision issues: {performance_feedback}

Rewrite to improve answer precision and accuracy:
1. Ask for specific numerical answers where applicable
2. Request units and formatting
3. Encourage double-checking calculations

Precise prompt:"""
        ]
        
        return meta_prompts
    
    def generate_prompt_variant(self, meta_prompt: str, max_length: int = 200) -> str:
        """
        Generate a prompt variant using meta-prompting
        
        Args:
            meta_prompt: Meta-prompt for generating the variant
            max_length: Maximum length of generated variant
            
        Returns:
            Generated prompt variant
        """
        try:
            # Tokenize meta-prompt
            inputs = self.tokenizer.encode(meta_prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # Generate with controlled parameters for prompt generation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,  # Moderate creativity for prompt generation
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode and extract the generated variant
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part (the prompt variant)
            variant = generated_text[len(meta_prompt):].strip()
            
            # Clean up the variant
            variant = self.clean_prompt_variant(variant)
            
            return variant
            
        except Exception as e:
            logger.error(f"Error generating prompt variant: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def clean_prompt_variant(self, variant: str) -> str:
        """
        Clean and format a generated prompt variant
        
        Args:
            variant: Raw generated prompt variant
            
        Returns:
            Cleaned prompt variant
        """
        # Remove common artifacts
        variant = variant.strip()
        
        # Remove quotes if the entire variant is quoted
        if variant.startswith('"') and variant.endswith('"'):
            variant = variant[1:-1]
        
        # Remove trailing punctuation that doesn't belong
        while variant.endswith(('.', '!', '?')) and len(variant) > 1:
            if variant[-2:] in ['?.', '!.', '..']:
                variant = variant[:-1]
            else:
                break
        
        # Limit length to reasonable size
        if len(variant) > 500:
            variant = variant[:500] + "..."
        
        # Ensure it's not empty
        if not variant.strip():
            variant = "Please solve this problem step by step."
        
        return variant
    
    def evaluate_prompt_performance(self, 
                                  prompt: str, 
                                  task_data: Dict[str, Any],
                                  tot_engine) -> Tuple[float, str]:
        """
        Evaluate the performance of a prompt using ToT reasoning
        
        Args:
            prompt: Prompt to evaluate
            task_data: Task information
            tot_engine: ToT engine for evaluation
            
        Returns:
            Tuple of (performance_score, feedback_text)
        """
        try:
            # Create a modified task with the new prompt
            eval_task = task_data.copy()
            eval_task["problem"] = f"{prompt}\n\nProblem: {task_data['problem']}"
            
            # Run ToT reasoning with the prompt
            result = tot_engine.reason(eval_task)
            
            # Calculate performance score based on multiple factors
            performance_score = 0.0
            feedback_parts = []
            
            # Factor 1: Consensus score (how well paths agree)
            consensus_weight = 0.4
            performance_score += result.consensus_score * consensus_weight
            feedback_parts.append(f"Consensus score: {result.consensus_score:.3f}")
            
            # Factor 2: Reasoning quality (average across paths)
            if result.reasoning_paths:
                avg_quality = sum(path.reasoning_quality for path in result.reasoning_paths) / len(result.reasoning_paths)
                quality_weight = 0.3
                performance_score += avg_quality * quality_weight
                feedback_parts.append(f"Average reasoning quality: {avg_quality:.3f}")
            
            # Factor 3: Answer coherence (simple heuristic)
            answer_coherence = self.assess_answer_coherence(result.final_answer, task_data)
            coherence_weight = 0.3
            performance_score += answer_coherence * coherence_weight
            feedback_parts.append(f"Answer coherence: {answer_coherence:.3f}")
            
            feedback_text = "; ".join(feedback_parts)
            
            return performance_score, feedback_text
            
        except Exception as e:
            logger.error(f"Error evaluating prompt performance: {e}")
            return 0.0, f"Evaluation error: {str(e)}"
    
    def assess_answer_coherence(self, answer: str, task_data: Dict[str, Any]) -> float:
        """
        Assess how coherent and relevant an answer is
        
        Args:
            answer: Generated answer
            task_data: Task information for context
            
        Returns:
            Coherence score between 0 and 1
        """
        if not answer or len(answer.strip()) < 3:
            return 0.0
        
        coherence_score = 0.5  # Base score
        
        # Check for domain-relevant terms
        domain = task_data.get("domain", "")
        domain_keywords = {
            "math": ["calculate", "equation", "number", "result", "answer", "km/h", "speed"],
            "logic": ["therefore", "assume", "if", "then", "conclusion"],
            "geometry": ["area", "triangle", "angle", "cm", "square"],
            "probability": ["probability", "chance", "outcome", "%"]
        }
        
        if domain in domain_keywords:
            answer_lower = answer.lower()
            keyword_count = sum(1 for keyword in domain_keywords[domain] if keyword in answer_lower)
            coherence_score += min(0.3, keyword_count * 0.1)
        
        # Check for numerical answers in math/science domains
        if domain in ["math", "geometry", "probability"]:
            import re
            numbers = re.findall(r'\d+\.?\d*', answer)
            if numbers:
                coherence_score += 0.2
        
        return min(1.0, coherence_score)
    
    def optimize_prompt(self, 
                       task_data: Dict[str, Any],
                       initial_prompt: str,
                       tot_engine) -> OptimizationResult:
        """
        Main optimization method - iteratively improve a prompt
        
        Args:
            task_data: Task information
            initial_prompt: Starting prompt to optimize
            tot_engine: ToT engine for evaluation
            
        Returns:
            OptimizationResult with best prompt and metrics
        """
        start_time = time.time()
        logger.info(f"Starting prompt optimization for task: {task_data['id']}")
        
        # Track all variants generated
        all_variants = []
        
        # Evaluate initial prompt
        initial_score, initial_feedback = self.evaluate_prompt_performance(
            initial_prompt, task_data, tot_engine
        )
        
        # Create initial variant record
        initial_variant = PromptVariant(
            id="initial",
            content=initial_prompt,
            version=0,
            parent_id=None,
            performance_score=initial_score,
            task_id=task_data["id"],
            generation_method="original",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata={"feedback": initial_feedback}
        )
        all_variants.append(initial_variant)
        
        # Track best prompt
        best_prompt = initial_prompt
        best_score = initial_score
        best_variant = initial_variant
        
        # Optimization loop
        current_prompt = initial_prompt
        current_score = initial_score
        current_feedback = initial_feedback
        
        for iteration in range(self.max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate meta-prompts for this iteration
            meta_prompts = self.create_meta_prompts(task_data, current_prompt, current_feedback)
            
            # Generate variants from meta-prompts
            iteration_variants = []
            
            for i, meta_prompt in enumerate(meta_prompts):
                variant_content = self.generate_prompt_variant(meta_prompt)
                
                # Evaluate the variant
                variant_score, variant_feedback = self.evaluate_prompt_performance(
                    variant_content, task_data, tot_engine
                )
                
                # Create variant record
                variant = PromptVariant(
                    id=f"iter_{iteration+1}_var_{i+1}",
                    content=variant_content,
                    version=iteration + 1,
                    parent_id=best_variant.id,
                    performance_score=variant_score,
                    task_id=task_data["id"],
                    generation_method=f"meta_prompt_{i+1}",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    metadata={"feedback": variant_feedback, "meta_prompt_type": i+1}
                )
                
                iteration_variants.append(variant)
                all_variants.append(variant)
                
                # Update best if this variant is better
                if variant_score > best_score:
                    best_prompt = variant_content
                    best_score = variant_score
                    best_variant = variant
                    logger.info(f"New best prompt found with score: {best_score:.3f}")
            
            # Check for improvement
            improvement = best_score - current_score
            if improvement < self.min_improvement_threshold:
                logger.info(f"Improvement below threshold ({improvement:.3f} < {self.min_improvement_threshold}), stopping optimization")
                break
            
            # Update current prompt for next iteration
            current_prompt = best_prompt
            current_score = best_score
            current_feedback = best_variant.metadata.get("feedback", "")
        
        execution_time = time.time() - start_time
        improvement_score = best_score - initial_score
        
        # Create optimization result
        result = OptimizationResult(
            task_id=task_data["id"],
            original_prompt=initial_prompt,
            best_prompt=best_prompt,
            improvement_score=improvement_score,
            optimization_iterations=iteration + 1,
            generated_variants=all_variants,
            execution_time=execution_time
        )
        
        logger.info(f"Prompt optimization completed in {execution_time:.2f}s")
        logger.info(f"Improvement score: {improvement_score:.3f}")
        logger.info(f"Best prompt score: {best_score:.3f}")
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Test with a simple task
    test_task = {
        "id": "test_opt",
        "domain": "math",
        "problem": "What is 25 + 17?",
        "expected_solution": "42"
    }
    
    initial_prompt = "Solve this problem:"
    
    # Initialize optimizer
    optimizer = PromptOptimizer(max_iterations=2)
    
    print("Prompt Optimizer initialized and ready for use")
    print(f"Test task: {test_task['problem']}")
    print(f"Initial prompt: {initial_prompt}")
