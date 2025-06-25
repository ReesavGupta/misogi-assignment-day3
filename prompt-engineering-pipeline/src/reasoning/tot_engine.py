"""
Tree-of-Thought (ToT) Reasoning Engine
Implements multi-path reasoning with Self-Consistency aggregation
"""

import json
import os
import time
import sys
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    AutoModelForCausalLM, AutoTokenizer
)
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from model_config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """Represents a single thought/reasoning step in the tree"""
    id: str
    content: str
    parent_id: Optional[str]
    depth: int
    confidence: float
    timestamp: str


@dataclass
class ReasoningPath:
    """Represents a complete reasoning path from root to conclusion"""
    path_id: str
    nodes: List[ThoughtNode]
    final_answer: str
    confidence_score: float
    reasoning_quality: float


@dataclass
class ToTResult:
    """Final result from Tree-of-Thought reasoning"""
    task_id: str
    final_answer: str
    reasoning_paths: List[ReasoningPath]
    consensus_score: float
    execution_time: float
    model_info: Dict[str, Any]


class ToTEngine:
    """Tree-of-Thought reasoning engine using GPT-2"""
    
    def __init__(self, model_name: str = "gpt2", max_depth: int = 4, num_paths: int = 6):
        """
        Initialize ToT Engine

        Args:
            model_name: HuggingFace model name (default: gpt2)
            max_depth: Maximum depth for reasoning tree (increased default to 4)
            num_paths: Number of reasoning paths to generate (increased default to 6)
        """
        self.model_name = model_name
        self.max_depth = max_depth
        self.num_paths = num_paths
        
        # Initialize model and tokenizer based on model type
        logger.info(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer based on model type
        if "gpt2" in model_name.lower():
            # GPT-2 models
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)
        else:
            # Modern models (Llama, Mistral, etc.) - use AutoTokenizer/AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Check available VRAM and use quantization if needed
            if self.device.type == "cuda":
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Available VRAM: {vram_gb:.1f}GB")

                if vram_gb < 8:
                    # Use float16 for low VRAM (more stable than 4-bit)
                    logger.info("Using float16 for low VRAM")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16
                    )
                else:
                    # Use standard loading for sufficient VRAM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
            else:
                # CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded on device: {self.device}")
        
    def generate_thought(self, prompt: str, max_length: int = 250, domain: str = "general") -> str:
        """
        Generate a single thought/reasoning step using GPT-2
        
        Args:
            prompt: Input prompt for generation
            max_length: Maximum length of generated text
            
        Returns:
            Generated thought text
        """
        try:
            # Tokenize input with proper attention mask
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 if "llama" in self.model_name.lower() else 512,
                padding=True
            )

            # Move to device
            if self.device.type == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)

            # Get optimized parameters for this domain
            gen_params = ModelConfig.get_optimized_generation_params(self.model_name, domain)

            # Adjust max_length for different models
            if "llama" in self.model_name.lower():
                max_new_tokens = min(max_length, 512)  # Llama can handle longer sequences
            elif "phi" in self.model_name.lower():
                max_new_tokens = min(max_length, 256)  # Phi-3 optimized length
            else:
                max_new_tokens = max_length

            # Generate with optimized parameters for reasoning
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        temperature=gen_params.get("temperature", 0.7),
                        top_p=gen_params.get("top_p", 0.85),
                        top_k=gen_params.get("top_k", 50),
                        do_sample=gen_params.get("do_sample", True),
                        repetition_penalty=gen_params.get("repetition_penalty", 1.1),
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False  # Disable cache to avoid compatibility issues
                    )
                except Exception as e:
                    logger.warning(f"Generation failed with cache, retrying without: {e}")
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                        temperature=gen_params.get("temperature", 0.7),
                        top_p=gen_params.get("top_p", 0.85),
                        top_k=gen_params.get("top_k", 50),
                        do_sample=gen_params.get("do_sample", True),
                        repetition_penalty=gen_params.get("repetition_penalty", 1.1),
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False
                    )
            
            # Decode and clean output
            if "llama" in self.model_name.lower():
                # For Llama models, extract only the new tokens
                input_length = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.shape[1]
                generated_tokens = outputs[0][input_length:]
                new_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            else:
                # For GPT-2 models
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_text = generated_text[len(prompt):].strip()

            return new_text
            
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def create_reasoning_prompts(self, task_data: Dict[str, Any]) -> List[str]:
        """
        Create diverse reasoning prompts for the same task with domain-specific templates

        Args:
            task_data: Task information dictionary

        Returns:
            List of different reasoning prompts
        """
        problem = task_data["problem"]
        domain = task_data["domain"]

        # Domain-specific prompt templates
        domain_templates = self.get_domain_templates(domain, problem)

        # General reasoning approaches
        general_prompts = [
            # Chain-of-thought approach
            f"Problem: {problem}\n\nI need to solve this step-by-step. Let me break it down:\n\nStep 1:",

            # Analytical approach with explicit reasoning
            f"Let me analyze this {domain} problem systematically:\n{problem}\n\nFirst, I'll identify the given information:\n-",

            # Question-driven approach
            f"Problem: {problem}\n\nTo solve this, I need to ask the right questions:\nQ1: What information am I given?\nA1:",

            # Method-focused approach
            f"Problem: {problem}\n\nFor this {domain} problem, I should use the appropriate method:\n1. Identify the problem type:\n2. Choose the method:\n3. Apply the method:",
        ]

        # Combine domain-specific and general prompts
        all_prompts = domain_templates + general_prompts

        return all_prompts[:self.num_paths]

    def get_domain_templates(self, domain: str, problem: str) -> List[str]:
        """
        Get domain-specific prompt templates

        Args:
            domain: Problem domain
            problem: Problem statement

        Returns:
            List of domain-specific prompts
        """
        templates = {
            "geometry": [
                f"Geometry Problem: {problem}\n\nFor this geometry problem, I need to:\n1. Identify the shapes and given measurements\n2. Determine what formulas to use\n3. Apply the formulas step by step\n\nGiven information:",
                f"Problem: {problem}\n\nThis is a geometry problem. Let me visualize it and identify the key elements:\n- Shape type:\n- Given measurements:\n- What to find:\n\nSolution approach:"
            ],
            "math": [
                f"Math Problem: {problem}\n\nTo solve this mathematical problem:\n1. Identify the type of calculation needed\n2. Set up the equation or formula\n3. Solve step by step\n4. Check the answer\n\nStep 1:",
                f"Problem: {problem}\n\nMathematical approach:\n- Given values:\n- Required calculation:\n- Formula to use:\n\nCalculation:"
            ],
            "logic": [
                f"Logic Problem: {problem}\n\nFor this logical reasoning problem:\n1. Identify the premises\n2. Determine the logical structure\n3. Apply logical rules\n4. Draw conclusions\n\nPremises:",
                f"Problem: {problem}\n\nLogical analysis:\n- What do we know for certain?\n- What are we trying to prove?\n- What logical steps are needed?\n\nReasoning:"
            ],
            "probability": [
                f"Probability Problem: {problem}\n\nFor this probability problem:\n1. Identify the sample space\n2. Determine favorable outcomes\n3. Apply probability formulas\n4. Calculate the result\n\nSample space:",
                f"Problem: {problem}\n\nProbability analysis:\n- Total possible outcomes:\n- Favorable outcomes:\n- Probability formula to use:\n\nCalculation:"
            ],
            "code": [
                f"Coding Problem: {problem}\n\nTo debug or solve this code problem:\n1. Understand what the code should do\n2. Identify the issue or requirement\n3. Trace through the logic\n4. Provide the solution\n\nCode analysis:",
                f"Problem: {problem}\n\nDebugging approach:\n- Expected behavior:\n- Current behavior:\n- Potential issues:\n- Solution:"
            ]
        }

        return templates.get(domain, [
            f"Problem: {problem}\n\nFor this {domain} problem, let me think systematically:\n1. Understand the problem\n2. Identify the approach\n3. Solve step by step\n\nAnalysis:"
        ])

    def build_reasoning_tree(self, task_data: Dict[str, Any]) -> List[ReasoningPath]:
        """
        Build multiple reasoning trees for a given task
        
        Args:
            task_data: Task information dictionary
            
        Returns:
            List of reasoning paths
        """
        logger.info(f"Building reasoning tree for task: {task_data['id']}")
        
        reasoning_paths = []
        initial_prompts = self.create_reasoning_prompts(task_data)
        
        for i, initial_prompt in enumerate(initial_prompts):
            path_id = f"path_{i+1}"
            nodes = []
            current_prompt = initial_prompt
            
            # Build reasoning chain for this path
            for depth in range(self.max_depth):
                node_id = f"{path_id}_node_{depth+1}"
                
                # Generate thought for current step with domain context
                thought_content = self.generate_thought(current_prompt, max_length=100, domain=task_data.get("domain", "general"))
                
                # Create thought node
                node = ThoughtNode(
                    id=node_id,
                    content=thought_content,
                    parent_id=nodes[-1].id if nodes else None,
                    depth=depth,
                    confidence=0.8,  # Placeholder - could be computed
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                nodes.append(node)
                
                # Update prompt for next iteration
                current_prompt += f" {thought_content}\n\nNext:"
                
                # Stop if we seem to have reached a conclusion
                if any(word in thought_content.lower() for word in ["therefore", "answer is", "conclusion", "result"]):
                    break
            
            # Extract final answer from the last node
            final_answer = self.extract_answer(nodes[-1].content, task_data)
            
            # Create reasoning path
            path = ReasoningPath(
                path_id=path_id,
                nodes=nodes,
                final_answer=final_answer,
                confidence_score=sum(node.confidence for node in nodes) / len(nodes),
                reasoning_quality=self.assess_reasoning_quality(nodes, task_data)
            )
            
            reasoning_paths.append(path)
            
        return reasoning_paths
    
    def extract_answer(self, text: str, task_data: Dict[str, Any]) -> str:
        """
        Extract the final answer from generated text with domain-specific logic

        Args:
            text: Generated reasoning text
            task_data: Task information for context

        Returns:
            Extracted answer string
        """
        domain = task_data.get("domain", "")
        text_lower = text.lower()

        # Domain-specific extraction patterns
        if domain == "geometry":
            return self.extract_geometry_answer(text, text_lower)
        elif domain == "math":
            return self.extract_math_answer(text, text_lower)
        elif domain == "probability":
            return self.extract_probability_answer(text, text_lower)
        elif domain == "logic":
            return self.extract_logic_answer(text, text_lower)
        else:
            return self.extract_general_answer(text, text_lower)

    def extract_geometry_answer(self, text: str, text_lower: str) -> str:
        """Extract answers from geometry problems"""
        import re

        # Look for area, perimeter, volume, length patterns
        patterns = [
            r"area[:\s]*([0-9]+\.?[0-9]*\s*cm²?)",
            r"hypotenuse[:\s]*([0-9]+\.?[0-9]*\s*cm?)",
            r"perimeter[:\s]*([0-9]+\.?[0-9]*\s*cm?)",
            r"volume[:\s]*([0-9]+\.?[0-9]*\s*cm³?)",
            r"([0-9]+\.?[0-9]*)\s*cm²?\s*and\s*([0-9]+\.?[0-9]*)\s*cm?"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(0)

        return self.extract_general_answer(text, text_lower)

    def extract_math_answer(self, text: str, text_lower: str) -> str:
        """Extract answers from math problems"""
        import re

        # Look for numerical answers
        patterns = [
            r"answer[:\s]*([0-9]+\.?[0-9]*)",
            r"result[:\s]*([0-9]+\.?[0-9]*)",
            r"equals?\s*([0-9]+\.?[0-9]*)",
            r"=\s*([0-9]+\.?[0-9]*)",
            r"([0-9]+\.?[0-9]*)\s*is\s*the\s*answer"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)

        return self.extract_general_answer(text, text_lower)

    def extract_probability_answer(self, text: str, text_lower: str) -> str:
        """Extract answers from probability problems"""
        import re

        # Look for probability expressions
        patterns = [
            r"probability[:\s]*([0-9]+\.?[0-9]*/?[0-9]*)",
            r"chance[:\s]*([0-9]+\.?[0-9]*%?)",
            r"([0-9]+\.?[0-9]*)\s*out\s*of\s*([0-9]+)",
            r"([0-9]+\.?[0-9]*)/([0-9]+\.?[0-9]*)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(0)

        return self.extract_general_answer(text, text_lower)

    def extract_logic_answer(self, text: str, text_lower: str) -> str:
        """Extract answers from logic problems"""
        # Look for logical conclusions
        logic_indicators = ["therefore", "conclusion", "thus", "hence", "so"]

        for indicator in logic_indicators:
            if indicator in text_lower:
                parts = text_lower.split(indicator, 1)
                if len(parts) > 1:
                    answer = parts[1].strip().split('.')[0].split('\n')[0]
                    return answer[:100]

        return self.extract_general_answer(text, text_lower)

    def extract_general_answer(self, text: str, text_lower: str) -> str:
        """General answer extraction fallback"""
        # Look for explicit answer statements
        answer_indicators = ["answer is", "result is", "solution is", "therefore", "equals", "="]

        for indicator in answer_indicators:
            if indicator in text_lower:
                parts = text_lower.split(indicator, 1)
                if len(parts) > 1:
                    potential_answer = parts[1].strip().split('.')[0].split('\n')[0]
                    return potential_answer[:50]

        # Fallback: return last meaningful sentence
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            return sentences[-1][:50]

        return text[:50]
    
    def assess_reasoning_quality(self, nodes: List[ThoughtNode], task_data: Dict[str, Any]) -> float:
        """
        Assess the quality of reasoning in a path
        
        Args:
            nodes: List of thought nodes in the path
            task_data: Task information
            
        Returns:
            Quality score between 0 and 1
        """
        # Simple heuristic-based quality assessment
        quality_score = 0.5  # Base score
        
        # Check for logical progression
        if len(nodes) >= 2:
            quality_score += 0.2
            
        # Check for domain-relevant keywords
        domain_keywords = {
            "math": ["calculate", "equation", "formula", "number"],
            "logic": ["assume", "therefore", "if", "then"],
            "geometry": ["area", "angle", "triangle", "circle"],
            "probability": ["chance", "probability", "outcome", "random"]
        }
        
        domain = task_data.get("domain", "")
        if domain in domain_keywords:
            text_content = " ".join(node.content.lower() for node in nodes)
            keyword_count = sum(1 for keyword in domain_keywords[domain] if keyword in text_content)
            quality_score += min(0.3, keyword_count * 0.1)
        
        return min(1.0, quality_score)

    def verify_answer(self, answer: str, task_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Verify the answer for consistency and reasonableness

        Args:
            answer: Generated answer
            task_data: Task information

        Returns:
            Tuple of (verified_answer, confidence_score)
        """
        domain = task_data.get("domain", "")
        confidence = 1.0

        # Domain-specific verification
        if domain == "geometry":
            answer, confidence = self.verify_geometry_answer(answer, task_data)
        elif domain == "math":
            answer, confidence = self.verify_math_answer(answer, task_data)
        elif domain == "probability":
            answer, confidence = self.verify_probability_answer(answer, task_data)

        # General reasonableness checks
        if not answer or len(answer.strip()) < 2:
            confidence *= 0.3

        # Check for obvious hallucination indicators
        hallucination_indicators = ["[generation error", "i don't know", "cannot determine", "unclear"]
        if any(indicator in answer.lower() for indicator in hallucination_indicators):
            confidence *= 0.5

        return answer, confidence

    def verify_geometry_answer(self, answer: str, task_data: Dict[str, Any]) -> Tuple[str, float]:
        """Verify geometry answers"""
        import re
        confidence = 1.0

        # Check if answer contains expected units
        if "cm" not in answer.lower() and "area" in task_data.get("problem", "").lower():
            confidence *= 0.7

        # Check for reasonable numerical values
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            try:
                values = [float(n) for n in numbers]
                # Check for unreasonably large or small values
                if any(v > 10000 or v < 0 for v in values):
                    confidence *= 0.6
            except ValueError:
                confidence *= 0.8

        return answer, confidence

    def verify_math_answer(self, answer: str, task_data: Dict[str, Any]) -> Tuple[str, float]:
        """Verify math answers"""
        import re
        confidence = 1.0

        # Check if answer contains a number
        if not re.search(r'\d', answer):
            confidence *= 0.5

        return answer, confidence

    def verify_probability_answer(self, answer: str, task_data: Dict[str, Any]) -> Tuple[str, float]:
        """Verify probability answers"""
        import re
        confidence = 1.0

        # Check for probability range (0-1 or 0-100%)
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            try:
                values = [float(n) for n in numbers]
                # Probability should be between 0 and 1, or 0 and 100 for percentages
                if "%" in answer:
                    if any(v > 100 or v < 0 for v in values):
                        confidence *= 0.4
                else:
                    if any(v > 1 or v < 0 for v in values):
                        confidence *= 0.4
            except ValueError:
                confidence *= 0.7

        return answer, confidence

    def apply_self_consistency(self, reasoning_paths: List[ReasoningPath]) -> Tuple[str, float]:
        """
        Apply Self-Consistency to select the best answer
        
        Args:
            reasoning_paths: List of reasoning paths
            
        Returns:
            Tuple of (best_answer, consensus_score)
        """
        if not reasoning_paths:
            return "No answer generated", 0.0
            
        # Group answers by similarity (simple string matching for now)
        answer_groups = {}
        
        for path in reasoning_paths:
            answer = path.final_answer.strip().lower()
            
            # Find similar existing answer or create new group
            matched = False
            for existing_answer in answer_groups:
                if self.answers_similar(answer, existing_answer):
                    answer_groups[existing_answer].append(path)
                    matched = True
                    break
                    
            if not matched:
                answer_groups[answer] = [path]
        
        # Find the answer group with highest combined confidence
        best_answer = ""
        best_score = 0.0
        
        for answer, paths in answer_groups.items():
            # Weight by both frequency and confidence
            group_score = len(paths) * sum(path.confidence_score for path in paths) / len(paths)
            
            if group_score > best_score:
                best_score = group_score
                best_answer = paths[0].final_answer  # Use original casing
        
        # Calculate consensus score
        consensus_score = best_score / len(reasoning_paths)
        
        return best_answer, consensus_score
    
    def answers_similar(self, answer1: str, answer2: str, threshold: float = 0.8) -> bool:
        """
        Check if two answers are similar
        
        Args:
            answer1, answer2: Answers to compare
            threshold: Similarity threshold
            
        Returns:
            True if answers are similar
        """
        # Simple similarity check - could be improved with more sophisticated methods
        if answer1 == answer2:
            return True
            
        # Check for common substrings
        words1 = set(answer1.split())
        words2 = set(answer2.split())
        
        if not words1 or not words2:
            return False
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        
        return jaccard_similarity >= threshold
    
    def reason(self, task_data: Dict[str, Any]) -> ToTResult:
        """
        Main reasoning method - orchestrates the entire ToT process
        
        Args:
            task_data: Task information dictionary
            
        Returns:
            ToTResult with final answer and reasoning paths
        """
        start_time = time.time()
        
        logger.info(f"Starting ToT reasoning for task: {task_data['id']}")
        
        # Build reasoning trees
        reasoning_paths = self.build_reasoning_tree(task_data)
        
        # Apply self-consistency
        final_answer, consensus_score = self.apply_self_consistency(reasoning_paths)

        # Verify the final answer
        verified_answer, verification_confidence = self.verify_answer(final_answer, task_data)

        # Adjust consensus score based on verification
        adjusted_consensus_score = consensus_score * verification_confidence

        execution_time = time.time() - start_time
        
        # Create result
        result = ToTResult(
            task_id=task_data["id"],
            final_answer=verified_answer,
            reasoning_paths=reasoning_paths,
            consensus_score=adjusted_consensus_score,
            execution_time=execution_time,
            model_info={
                "model_name": self.model_name,
                "device": str(self.device),
                "num_paths": self.num_paths,
                "max_depth": self.max_depth,
                "verification_confidence": verification_confidence
            }
        )
        
        logger.info(f"ToT reasoning completed in {execution_time:.2f}s")
        logger.info(f"Final answer: {final_answer}")
        logger.info(f"Consensus score: {consensus_score:.3f}")
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Test with a simple task
    test_task = {
        "id": "test_001",
        "domain": "math",
        "problem": "What is 15 + 27?",
        "expected_solution": "42"
    }
    
    # Initialize engine
    engine = ToTEngine(num_paths=3, max_depth=2)
    
    # Run reasoning
    result = engine.reason(test_task)
    
    # Display results
    print(f"Task: {result.task_id}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Consensus Score: {result.consensus_score:.3f}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Number of Paths: {len(result.reasoning_paths)}")
