# Math Tutor AI - An Educational Assistant

I've been working on this math tutoring system that helps middle and high school students (grades 6-10) with their math homework. What makes this project interesting is that I've implemented four different ways for the AI to approach problems, and I wanted to see which one works best for teaching math.

## Getting Started

If you want to try this out, here's what you need to do:

```bash
# First, let's get everything set up
python setup.py

# Make sure everything's working properly
python test_setup.py

# Now you can start the math tutor
cd src/
python main.py
```

## What This Project Does

I built this as a command-line math tutor that runs locally using the Llama 3 8B model through Ollama. The interesting part is that I've implemented four different teaching approaches:

- **Direct instruction** - Just giving straightforward explanations
- **Learning from examples** - Showing similar problems first, then solving the new one
- **Step-by-step thinking** - Breaking down the reasoning process explicitly
- **Self-reflection** - Having the AI think about its own teaching approach

## How Everything's Organized

Here's how I've structured the project:

```
edtech-math-tutor/
├── README.md                  # What you're reading now
├── domain_analysis.md        # My research on math education
├── prompts/                  # The four different teaching approaches
│   ├── zero_shot.txt         # Direct instruction style
│   ├── few_shot.txt          # Example-based teaching
│   ├── cot_prompt.txt        # Step-by-step reasoning
│   └── meta_prompt.txt       # Self-reflective approach
├── evaluation/               # Testing and analysis
│   ├── input_queries.json    # Test questions I use
│   ├── output_logs.json      # Results from testing
│   └── analysis_report.md    # What I learned
├── src/                      # The actual code
│   ├── main.py              # Main program
│   └── utils.py             # Helper functions
└── hallucination_log.md     # When things go wrong
```

## Setting Everything Up

### The Easy Way
I've made this pretty straightforward. Just run:

```bash
cd edtech-math-tutor/
python setup.py
```

This script will handle most of the setup for you.

### If You Want to Do It Manually
Sometimes the automated setup doesn't work perfectly, so here's how to do it step by step:

1. **Get Ollama running**: Go to https://ollama.ai/ and install it for your operating system
2. **Download the AI model**:
   ```bash
   ollama pull llama3:8b
   ```
3. **Install the Python stuff**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Make sure Ollama is running** (it usually starts automatically, but just in case):
   ```bash
   ollama serve
   ```

### Actually Using the Tutor
Once everything's set up:

```bash
cd src/
python main.py
```

### Making Sure It Works
I've included a test script to check if everything's working:

```bash
python test_setup.py
```

### When Things Don't Work
Here are the most common issues I've run into:

- **Can't connect to Ollama**: Make sure it's actually running with `ollama serve`
- **Model not found**: You probably need to download it first: `ollama pull llama3:8b`
- **Permission problems**: Try using `python3` instead of `python`
- **Other weird issues**: Run the test script to see what's going wrong: `python test_setup.py`

## How I Evaluate the Different Approaches

I wanted to be systematic about testing which teaching method works best, so I rate each response on a scale of 1-5 for:

- **Getting the math right**: Is the answer actually correct?
- **Clear explanations**: Can a student follow the reasoning?
- **Avoiding nonsense**: Does it make up facts or give wrong information?
- **Being consistent**: Does it give similar quality answers to similar questions?

## What This Tutor Focuses On

I designed this specifically for middle and high school students (grades 6-10) who need help with:

1. Understanding math concepts like algebra, geometry, and basic arithmetic
2. Working through problems step by step
3. Getting extra practice problems to work on

## What I've Learned So Far

I'm still collecting data, but you can run the evaluation mode to see how the different approaches compare. The results get saved automatically so you can see which method works best for different types of problems.

## What's Actually Built

I've put together a complete system that includes:

**Four Different Teaching Styles**
- Direct answers (zero-shot)
- Learning from examples (few-shot)
- Thinking out loud (chain-of-thought)
- Self-aware teaching (meta-prompting)

**Easy-to-Use Interface**
The whole thing runs in your terminal, so no complicated setup or web browsers needed.

**Systematic Testing**
I can automatically test all four approaches on the same set of problems and compare how they do.

**Detailed Logging**
Everything gets saved so you can go back and see exactly what happened during each tutoring session.

**Automatic Reports**
The system generates summaries of how well each approach is working.

**Error Tracking**
When the AI says something wrong or confusing, I keep track of it to understand the patterns.

**Technical Stuff**
- Runs locally using Ollama and Llama 3 8B (so your data stays private)
- Each teaching approach is in its own file, so it's easy to modify them
- Handles connection problems gracefully
- Includes tools to verify everything's working correctly

**Testing Framework**
- Uses real math problems that cover grades 6-10
- Compares all four approaches side by side
- Generates detailed reports about what's working and what isn't

## How to Actually Use This

### Talking to the Tutor
When you start the program, choose option 1 for interactive mode. Then you can ask questions like:

```bash
python main.py
# Pick option 1

# Try asking things like:
"Explain how to solve 2x + 5 = 15"
"What is the area of a triangle?"
"Can you make up a practice problem for quadratic equations?"
"I got x = 2 when solving x/3 = 6. Is that right?"
"What's the difference between area and perimeter?"
```

### Testing All the Approaches
If you want to see how the different teaching methods compare:

```bash
python main.py
# Pick option 2
# This runs the same test questions through all four approaches
```

### Commands You Can Use
While you're chatting with the tutor, you can type:

- `help` - See what commands are available
- `strategy` - Switch between teaching styles (zero_shot, few_shot, cot_prompt, meta_prompt)
- `menu` - Go back to the main menu
- `quit` - Close the program
