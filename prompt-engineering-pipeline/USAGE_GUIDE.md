# How to Use the Prompt Engineering System

## Getting Started (Takes About 5 Minutes)

### **First, Make Sure Everything's Working**
```bash
# Navigate to the project and make sure your virtual environment is active
cd prompt-engineering-pipeline
# You should see (.venv) at the beginning of your command prompt

# Test that the basic components are working
python src/task_loader.py
```

### **Try Your First Reasoning Task**
```bash
# Let's run a simple math problem with multiple reasoning paths
python src/reasoning/runner.py --task math_001 --paths 2 --depth 2
```

### **See How Well It Performed**
```bash
# Generate an evaluation report for the task you just ran
python src/evaluation/evaluation_runner.py --task math_001
```

### **Check Out the Results**
```bash
# Look at what reports were generated
ls evaluation/
cat evaluation/report_*.md
```

## More Detailed Instructions

### **Running Specific Problems**

#### **Basic Usage**
```bash
# Run any task by its ID
python src/reasoning/runner.py --task TASK_ID

# Here are the tasks I've set up:
# - math_001: Calculating average speed (good starter problem)
# - logic_001: Knights and knaves logic puzzle
# - code_001: Finding bugs in Python code
# - word_001: Age-based word problem
# - pattern_001: Number sequence recognition
# - geometry_001: Triangle area calculation
# - probability_001: Basic coin flip probability
```

#### **More Advanced Options**
```bash
# Run with custom settings
python src/reasoning/runner.py --task math_001 --paths 3 --depth 3 --model gpt2

# Run all math problems at once
python src/reasoning/runner.py --domain math

# Run everything (this takes a while!)
python src/reasoning/runner.py --all
```

### **Making Prompts Better Automatically**

#### **Improve One Task**
```bash
# Let the system optimize prompts for a specific problem
python src/prompt_optimization/optimization_runner.py --task geometry_001 --iterations 3
```

#### **Improve All Tasks in a Category**
```bash
# Optimize all math tasks together
python src/prompt_optimization/optimization_runner.py --domain math --iterations 2
```

#### **Full Optimization Run**
```bash
# Run the complete optimization cycle
python src/prompt_optimization/optimization_runner.py --cycle --threshold 0.1
```

### **Analyzing Performance**

#### **Check One Task**
```bash
# See how well a specific task performed
python src/evaluation/evaluation_runner.py --task code_001
```

#### **Comprehensive Analysis**
```bash
# Evaluate everything
python src/evaluation/evaluation_runner.py --all

# Just look at logic problems
python src/evaluation/evaluation_runner.py --domain logic
```

#### **Full Analysis Cycle**
```bash
# Run complete evaluation with detailed insights
python src/evaluation/evaluation_runner.py --cycle
```

#### **Generate Insights**
```bash
# Create analysis from existing results
python src/evaluation/evaluation_runner.py --reflect
```

## Different Ways to Use This System

### **If You're Just Getting Started**
```bash
# 1. Pick one task to understand how everything works
python src/reasoning/runner.py --task math_001 --paths 2

# 2. See how well it performed
python src/evaluation/evaluation_runner.py --task math_001

# 3. Read the results
cat evaluation/report_*.md

# 4. Try making the prompt better
python src/prompt_optimization/optimization_runner.py --task math_001 --iterations 2

# 5. Check if the optimization actually helped
python src/evaluation/evaluation_runner.py --task math_001
```

### **If You Want to Run Everything**
```bash
# 1. Get baseline performance on all tasks
python src/evaluation/evaluation_runner.py --all

# 2. Let the system optimize all prompts
python src/prompt_optimization/optimization_runner.py --all --iterations 3

# 3. See how much everything improved
python src/evaluation/evaluation_runner.py --all

# 4. Generate insights about what worked
python src/evaluation/evaluation_runner.py --reflect

# 5. Read the analysis
cat evaluation/reflection.md
```

### **If You Want to Focus on One Type of Problem**
```bash
# 1. Run all math problems (or whatever domain you're interested in)
python src/reasoning/runner.py --domain math

# 2. See how the math tasks performed
python src/evaluation/evaluation_runner.py --domain math

# 3. Optimize just the math prompts
python src/prompt_optimization/optimization_runner.py --domain math

# 4. Check the improvements
python src/evaluation/evaluation_runner.py --domain math
```

## Understanding What Gets Generated

### **Reasoning Results** (in `logs/reasoning_paths/`)
- Files like `task_id_TIMESTAMP.json` contain detailed records of how the AI thought through each problem
- You'll see all the different reasoning paths, how much they agreed with each other, and how long everything took

### **Optimization History** (in `logs/optimization/`)
- Files like `task_id_optimization_TIMESTAMP.json` show how prompts evolved over time
- You can see different prompt variations, their performance scores, and improvement trends

### **Evaluation Reports** (in `evaluation/`)
- `report_TIMESTAMP.json` has all the detailed evaluation data
- `report_TIMESTAMP.md` is the human-readable version of the same information
- `reflection.md` contains my analysis and insights about what's working and what isn't

### **Prompt Evolution** (in `prompts/`)
- Files like `task_id_v1.txt`, `task_id_v2.txt` show how prompts improved over time
- `prompt_registry.json` keeps track of all the different versions

## Customizing How It Runs

### **Model and Reasoning Settings**
```bash
# Use a different model if you have it available
--model gpt2-medium

# Control how many different approaches it tries
--paths 5          # Number of reasoning paths (I usually use 3)
--depth 3          # How deep each reasoning chain goes (I usually use 2)
```

### **Optimization Settings**
```bash
--iterations 5     # How many times to try improving prompts (I usually use 3)
--threshold 0.05   # How much improvement is needed to keep going (I usually use 0.1)
```

## Checking That Everything's Working

### **Basic System Check**
```bash
# Make sure all the components are working
python src/task_loader.py
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### **Looking at the Logs**
```bash
# See what reasoning results have been generated
ls -la logs/reasoning_paths/
cat logs/reasoning_paths/math_001_*.json | head -20

# Check optimization history
ls -la logs/optimization/
```

### **Performance Monitoring**
```bash
# Quick check on how well a task is performing
python src/evaluation/evaluation_runner.py --task geometry_001
grep "Execution Time" evaluation/report_*.md
```

## When Things Go Wrong

### **Common Problems and Solutions**

1. **"Model loading error"**
   ```bash
   # Make sure you have internet for the first download
   python -c "from transformers import GPT2LMHeadModel; print('Model accessible')"
   ```

2. **"CUDA out of memory"**
   ```bash
   # The system usually handles this automatically, but you can force CPU mode:
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **"Task not found"**
   ```bash
   # See what tasks are actually available
   python -c "from src.task_loader import TaskLoader; loader = TaskLoader(); print(loader.get_task_summary())"
   ```

## Getting Better Performance

### **If You Want Things to Run Faster**
- Use fewer reasoning paths: `--paths 2`
- Keep reasoning shallow: `--depth 2`
- Don't optimize as much: `--iterations 2`

### **If You Want Better Quality Results**
- Try more reasoning paths: `--paths 5`
- Go deeper in reasoning: `--depth 3`
- Optimize more thoroughly: `--iterations 5`

### **If You're Running Low on Memory**
- I've already optimized this for 4GB VRAM
- The system automatically falls back to CPU if needed
- Batch processing is designed to be memory-efficient

