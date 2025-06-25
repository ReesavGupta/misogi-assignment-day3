# Project Requirements - Advanced Prompt Engineering System

**What I'm Building:** A system that uses multiple reasoning paths and automatically improves its own prompts
**Who's Working on This:** Me (with some help from AI tools)
**Last Updated:** June 25, 2025

## What I'm Trying to Achieve

I want to build a prompt engineering system that goes way beyond the typical "ask once, get one answer" approach. Here's what I'm aiming for:

1. **Smart multi-path reasoning** - Using Tree-of-Thought and self-consistency to explore different ways of solving problems
2. **Self-improving prompts** - Taking inspiration from OPRO and TextGrad to automatically make prompts better over time
3. **Focus on complex reasoning** - Targeting challenging tasks like multi-step math problems, logic puzzles, and code debugging

## The Core Challenge

When I give this system a complex reasoning task, here's what should happen:

1. Generate several different approaches to solving the problem using a local GPT-2 model
2. Compare the different answers and pick the most consistent/reliable one
3. Figure out where the system is struggling
4. Automatically create improved prompt variations
5. Let me review any weird or wrong outputs and help refine the approach

## What I Need to Build

### Part 1: Setting Up Test Problems ✅

**What I need to do:** Collect 5-7 different reasoning challenges that will really test the system.

**How I'm organizing this:**
- Save everything in the `tasks/` folder as JSON or YAML files
- Each problem needs:
  - A unique ID so I can track it
  - What domain it's from (math, logic, coding, etc.)
  - The actual problem statement
  - What the correct answer should be
  - How to check if an answer is right

**Here's what a task file looks like:**
```json
{
  "id": "math_001",
  "domain": "math",
  "problem": "A train travels 60 km/h for 1.5 hours and 90 km/h for 2 hours. What is the average speed?",
  "expected_solution": "81 km/h",
  "evaluation_rule": "exact_match"
}
```

### Part 2: Multi-Path Reasoning Engine ✅

**What I need to do:** Build the core system that generates multiple ways to solve each problem and picks the best answer.

**How this works:**
- Put the code in `src/reasoning/`
- For each problem:
  - Generate 3-5 completely different approaches to solving it
  - Organize each approach as a tree structure where each "thought" is a node and different logical paths are branches
  - Remove approaches that are obviously wrong or just repeating others
  - Use self-consistency to pick the final answer by seeing which approaches agree
  - Keep detailed logs of everything

**What the output looks like:**
```json
{
  "task_id": "math_001",
  "final_answer": "81 km/h",
  "reasoning_paths": [...],
  "consensus_score": 0.67
}
```

- All results get saved to `logs/reasoning_paths/{task_id}.json`

### Part 3: Self-Improving Prompts ✅

**What I need to do:** Build a system that automatically makes prompts better when they're not working well.

**How this works:**
- Code goes in `src/prompt_optimization/`
- When a task isn't performing well:
  - The system automatically starts an improvement loop
  - It uses meta-prompting to create 3-5 different versions of the original prompt
  - For example, it might say: *"The original prompt failed on Task X. Rewrite it to be clearer, more detailed, or better at encouraging step-by-step reasoning."*
  - New prompt versions get saved as `prompts/{task_id}_v{n}.txt`

**Important requirements:**
- Every iteration gets a version number so I can track the evolution
- The optimizer creates multiple alternatives, not just one
- All optimization results get logged to `logs/optimization/{task_id}.json`

### Part 4: Measuring Success ✅

**What I need to do:** Build a comprehensive evaluation system to see how well everything is working.

**Where this goes:** `evaluation/` folder

**What I'm measuring:**
| What I'm Tracking | What It Means |
|--------|-------------|
| Task Accuracy | How often does it get the right answer? |
| Reasoning Quality | Do the logical steps make sense? (I'll review this manually or use another AI to help) |
| Nonsense Rate | How often does it make unsupported claims or logical errors? |
| Improvement Over Time | Do the optimized prompts actually work better than the originals? |

**What I need to produce:**
- `evaluation/results.json` - Detailed scores for each task
- `evaluation/metrics_report.md` - Summary of overall performance
- `evaluation/reflection.md` - My analysis of what worked well and what didn't

## How the Whole System Works

Here's the step-by-step process:

1. **Load a problem** from the `tasks/` folder
2. **Generate multiple reasoning approaches** using Tree-of-Thought with GPT-2, save everything to `logs/`
3. **Pick the best answer** using self-consistency (see which approaches agree)
4. **Check how well it performed**
   - If it's not doing well, automatically trigger the prompt optimizer to create better versions
5. **Test the new prompts** by running steps 2-4 again
6. **Keep detailed records** of all attempts and prompt versions

## How I'm Organizing the Files

```
prompt-engineering-pipeline/
 ┣ tasks/                       # All the test problems
 ┃ ┗ task_*.json               # Individual problem files
 ┣ prompts/                     # Different prompt versions
 ┃ ┣ task_001_v1.txt           # Original prompt for task 1
 ┃ ┣ task_001_v2.txt           # Improved version
 ┣ src/                         # The actual code
 ┃ ┣ reasoning/                 # Tree-of-Thought implementation
 ┃ ┣ prompt_optimization/       # Automatic prompt improvement
 ┣ logs/                        # All the results and debugging info
 ┃ ┣ reasoning_paths/           # Records of different thinking approaches
 ┃ ┣ optimization/              # How prompts evolved
 ┣ evaluation/                  # Performance analysis
 ┃ ┣ results.json              # Detailed scores
 ┃ ┣ metrics_report.md         # Summary report
 ┃ ┗ reflection.md             # My analysis of what worked
 ┗ README.md                   # Overview and instructions
```

## Technical Constraints and Choices

**What I'm using:**
- **Local AI model:** GPT-2 through HuggingFace Transformers (keeps everything private)
- **Hardware:** NVIDIA RTX 2050 with 4GB VRAM (not the most powerful, but it works)
- **Batch size:** Keeping it small (1-2 items at a time) to avoid running out of memory
- **Required packages:** transformers, datasets, accelerate, tqdm, pandas

## How I'll Know This Is Successful

1. **Everything works:** All four parts of the system are implemented and functional
2. **Actually improves:** The optimized prompts demonstrably perform better than the originals
3. **Handles variety:** Can successfully work with 5-7 different types of reasoning tasks
4. **Well documented:** Clear logs, metrics, and analysis reports
5. **Reproducible:** Anyone can run the same experiments and get similar results

## My Development Plan

1. Set up the folder structure and basic framework
2. Build the task curation system
3. Implement the Tree-of-Thought reasoning engine
4. Create the automatic prompt optimization system
5. Build the evaluation and analysis framework
6. Run comprehensive end-to-end tests
7. Generate final reports and documentation
