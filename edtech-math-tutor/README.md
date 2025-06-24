# EdTech Math Tutor - Prompt Engineering Lab

A domain-specific LLM agent that serves as a math tutor for grades 6-10, implementing and evaluating 4 different prompt engineering strategies.

## 🚀 Quick Start

```bash
# 1. Setup the project
python setup.py

# 2. Validate installation
python test_setup.py

# 3. Start the math tutor
cd src/
python main.py
```

## 🎯 Project Overview

This project builds a CLI-based math tutor using Ollama with Llama 3 8B model, comparing different prompt engineering approaches:
- Zero-shot prompting
- Few-shot prompting  
- Chain-of-Thought (CoT) prompting
- Meta-prompting

## 🏗️ Project Structure

```
edtech-math-tutor/
├── README.md                  # This file
├── domain_analysis.md        # Domain understanding & tasks
├── prompts/                  # Prompt strategies
│   ├── zero_shot.txt
│   ├── few_shot.txt
│   ├── cot_prompt.txt
│   └── meta_prompt.txt
├── evaluation/               # Evaluation framework
│   ├── input_queries.json
│   ├── output_logs.json
│   ├── analysis_report.md
├── src/                     # Source code
│   ├── main.py
│   └── utils.py
└── hallucination_log.md     # Track failure cases
```

## 🚀 Setup Instructions

### Quick Setup (Recommended)
```bash
# Clone or download the project
cd edtech-math-tutor/

# Run the automated setup script
python setup.py
```

### Manual Setup
1. **Install Ollama**: Visit https://ollama.ai/ and install for your OS
2. **Pull Llama 3 8B model**:
   ```bash
   ollama pull llama3:8b
   ```
3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start Ollama service** (if not auto-started):
   ```bash
   ollama serve
   ```

### Running the Math Tutor
```bash
cd src/
python main.py
```

### Validation
Test your setup with the validation script:
```bash
python test_setup.py
```

### Validation
Test your setup with the validation script:
```bash
python test_setup.py
```

### Troubleshooting
- **Ollama connection error**: Ensure Ollama is running (`ollama serve`)
- **Model not found**: Run `ollama pull llama3:8b` to download the model
- **Permission errors**: Try running with `python3` instead of `python`
- **Setup issues**: Run `python test_setup.py` to diagnose problems
- **Setup issues**: Run `python test_setup.py` to diagnose problems

## 📊 Evaluation Metrics

Each response is manually rated on a 1-5 scale for:
- **Accuracy**: Mathematical correctness
- **Reasoning Clarity**: Step-by-step explanation quality  
- **Hallucinations**: Factual errors or made-up information
- **Consistency**: Stable performance across runs

## 🎓 Domain Focus

**Target**: Math tutor for grades 6-10
**Core Tasks**:
1. Explain math concepts (algebra, geometry, arithmetic)
2. Solve step-by-step problems
3. Generate practice problems

## 📈 Key Findings

[To be filled after evaluation - use the evaluation mode and report generation features]

## 🎯 Project Features

### ✅ Complete Implementation
- **4 Prompt Strategies**: Zero-shot, Few-shot, Chain-of-Thought, Meta-prompting
- **CLI Interface**: User-friendly command-line interaction
- **Systematic Evaluation**: Automated testing with manual rating system
- **Comprehensive Logging**: All interactions saved for analysis
- **Report Generation**: Automated analysis and summary reports
- **Error Tracking**: Dedicated hallucination logging system

### 🔧 Technical Components
- **Ollama Integration**: Local LLM inference with Llama 3 8B
- **Modular Design**: Separate prompt files for easy modification
- **Robust Error Handling**: Connection checks and fallback mechanisms
- **Validation System**: Setup verification and testing tools

### 📊 Evaluation Framework
- **Manual Rating System**: 1-5 scale across 4 criteria
- **Structured Test Cases**: 10 diverse math problems covering grades 6-10
- **Performance Comparison**: Side-by-side strategy analysis
- **Detailed Reporting**: Automated insights and recommendations

## 🔧 Usage Examples

### Interactive Mode
```bash
# Start the tutor
python main.py
# Select option 1 for Interactive Mode

# Example student queries:
"Explain how to solve 2x + 5 = 15"
"What is the area of a triangle?"
"Generate a practice problem for quadratic equations"
"I got x = 2 when solving x/3 = 6. Is this correct?"
"What's the difference between area and perimeter?"
```

### Evaluation Mode
```bash
# Run systematic evaluation
python main.py
# Select option 2 for Evaluation Mode
# This will test all 4 prompt strategies on predefined queries
```

### Available Commands (in Interactive Mode)
- `help` - Show available commands
- `strategy` - Change prompt strategy (zero_shot, few_shot, cot_prompt, meta_prompt)
- `menu` - Return to main menu
- `quit` - Exit the application
