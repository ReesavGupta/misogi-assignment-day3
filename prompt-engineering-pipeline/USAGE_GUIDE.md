# 🚀 Prompt Engineering Pipeline - Usage Guide

## 🏁 Quick Start (5 Minutes)

### **Step 1: Verify Setup**
```bash
# Make sure you're in the project directory and venv is active
cd prompt-engineering-pipeline
# You should see (.venv) in your prompt

# Test basic functionality
python src/task_loader.py
```

### **Step 2: Run Your First Task**
```bash
# Run a simple math task with Tree-of-Thought reasoning
python src/reasoning/runner.py --task math_001 --paths 2 --depth 2
```

### **Step 3: Evaluate Performance**
```bash
# Evaluate the task and generate a report
python src/evaluation/evaluation_runner.py --task math_001
```

### **Step 4: View Results**
```bash
# Check the generated reports
ls evaluation/
cat evaluation/report_*.md
```

## 📚 **Detailed Usage Guide**

### **🎯 1. Running Individual Tasks**

#### **Basic Task Execution**
```bash
# Run a specific task
python src/reasoning/runner.py --task TASK_ID

# Available tasks:
# - math_001: Average speed calculation
# - logic_001: Knights and knaves puzzle  
# - code_001: Python debugging
# - word_001: Age puzzle
# - pattern_001: Number sequence
# - geometry_001: Triangle area
# - probability_001: Coin flip probability
```

#### **Customized Execution**
```bash
# Run with custom parameters
python src/reasoning/runner.py --task math_001 --paths 3 --depth 3 --model gpt2

# Run all tasks in a domain
python src/reasoning/runner.py --domain math

# Run all tasks
python src/reasoning/runner.py --all
```

### **🔧 2. Prompt Optimization**

#### **Optimize Single Task**
```bash
# Optimize prompts for a specific task
python src/prompt_optimization/optimization_runner.py --task geometry_001 --iterations 3
```

#### **Domain-Wide Optimization**
```bash
# Optimize all tasks in a domain
python src/prompt_optimization/optimization_runner.py --domain math --iterations 2
```

#### **Full Optimization Cycle**
```bash
# Run complete optimization cycle
python src/prompt_optimization/optimization_runner.py --cycle --threshold 0.1
```

### **📊 3. Evaluation & Analysis**

#### **Single Task Evaluation**
```bash
# Evaluate one task
python src/evaluation/evaluation_runner.py --task code_001
```

#### **Comprehensive Evaluation**
```bash
# Evaluate all tasks
python src/evaluation/evaluation_runner.py --all

# Evaluate by domain
python src/evaluation/evaluation_runner.py --domain logic
```

#### **Complete Evaluation Cycle**
```bash
# Run full evaluation with reflection
python src/evaluation/evaluation_runner.py --cycle
```

#### **Reflection Analysis**
```bash
# Generate insights from existing reports
python src/evaluation/evaluation_runner.py --reflect
```

## 🎮 **Recommended Workflows**

### **🔬 Workflow 1: Research & Experimentation**
```bash
# 1. Start with a single task to understand the system
python src/reasoning/runner.py --task math_001 --paths 2

# 2. Evaluate the results
python src/evaluation/evaluation_runner.py --task math_001

# 3. Check the evaluation report
cat evaluation/report_*.md

# 4. Try optimizing the prompt
python src/prompt_optimization/optimization_runner.py --task math_001 --iterations 2

# 5. Re-evaluate to see improvement
python src/evaluation/evaluation_runner.py --task math_001
```

### **🏭 Workflow 2: Production Pipeline**
```bash
# 1. Run complete evaluation baseline
python src/evaluation/evaluation_runner.py --all

# 2. Run optimization cycle
python src/prompt_optimization/optimization_runner.py --all --iterations 3

# 3. Re-evaluate after optimization
python src/evaluation/evaluation_runner.py --all

# 4. Generate reflection insights
python src/evaluation/evaluation_runner.py --reflect

# 5. Review recommendations
cat evaluation/reflection.md
```

### **🎯 Workflow 3: Domain-Specific Focus**
```bash
# 1. Focus on a specific domain (e.g., math)
python src/reasoning/runner.py --domain math

# 2. Evaluate domain performance
python src/evaluation/evaluation_runner.py --domain math

# 3. Optimize domain-specific prompts
python src/prompt_optimization/optimization_runner.py --domain math

# 4. Re-evaluate improvements
python src/evaluation/evaluation_runner.py --domain math
```

## 📁 **Understanding Output Files**

### **Reasoning Results** (`logs/reasoning_paths/`)
- `task_id_TIMESTAMP.json`: Detailed reasoning traces
- Contains all reasoning paths, consensus scores, execution times

### **Optimization Logs** (`logs/optimization/`)
- `task_id_optimization_TIMESTAMP.json`: Optimization history
- Shows prompt variants, performance scores, improvement trends

### **Evaluation Reports** (`evaluation/`)
- `report_TIMESTAMP.json`: Detailed evaluation data
- `report_TIMESTAMP.md`: Human-readable evaluation report
- `reflection.md`: Latest reflection analysis with insights

### **Prompt Versions** (`prompts/`)
- `task_id_v1.txt`, `task_id_v2.txt`: Prompt versions
- `prompt_registry.json`: Prompt version tracking

## ⚙️ **Configuration Options**

### **Model Settings**
```bash
# Use different model (if available)
--model gpt2-medium

# Adjust reasoning parameters
--paths 5          # Number of reasoning paths (default: 3)
--depth 3          # Maximum reasoning depth (default: 2)
```

### **Optimization Settings**
```bash
--iterations 5     # Max optimization iterations (default: 3)
--threshold 0.05   # Min improvement threshold (default: 0.1)
```

## 🔍 **Monitoring & Debugging**

### **Check System Status**
```bash
# Verify all components
python src/task_loader.py
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### **View Logs**
```bash
# Check recent reasoning results
ls -la logs/reasoning_paths/
cat logs/reasoning_paths/math_001_*.json | head -20

# Check optimization logs
ls -la logs/optimization/
```

### **Performance Monitoring**
```bash
# Quick performance check
python src/evaluation/evaluation_runner.py --task geometry_001
grep "Execution Time" evaluation/report_*.md
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **"Model loading error"**
   ```bash
   # Check internet connection for first-time model download
   python -c "from transformers import GPT2LMHeadModel; print('Model accessible')"
   ```

2. **"CUDA out of memory"**
   ```bash
   # Reduce batch size or use CPU
   # The system automatically handles this, but you can force CPU:
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **"Task not found"**
   ```bash
   # List available tasks
   python -c "from src.task_loader import TaskLoader; loader = TaskLoader(); print(loader.get_task_summary())"
   ```

## 📈 **Performance Tips**

### **For Faster Execution**
- Use fewer reasoning paths: `--paths 2`
- Reduce reasoning depth: `--depth 2`
- Limit optimization iterations: `--iterations 2`

### **For Better Quality**
- Increase reasoning paths: `--paths 5`
- Increase reasoning depth: `--depth 3`
- More optimization iterations: `--iterations 5`

### **For Memory Efficiency**
- The system is already optimized for 4GB VRAM
- CPU fallback is automatic
- Batch processing is memory-efficient

## 🎯 **Next Steps**

1. **Start Simple**: Run a single task evaluation
2. **Explore**: Try different domains and tasks
3. **Optimize**: Use the optimization pipeline
4. **Analyze**: Review evaluation reports and reflections
5. **Iterate**: Use insights to improve performance

## 📞 **Getting Help**

- Check `PROJECT_COMPLETION_SUMMARY.md` for technical details
- Review `PRD.md` for system specifications
- Examine log files for detailed execution traces
- All components have built-in help: `python script.py --help`

---

**Ready to start? Try the Quick Start section above! 🚀**
