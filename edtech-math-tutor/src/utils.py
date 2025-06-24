"""
Utility functions for the EdTech Math Tutor
"""

import json
import os
from datetime import datetime

def load_prompt(strategy):
    """Load prompt template for given strategy"""
    prompt_file = f"../prompts/{strategy}.txt"
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: Prompt file '{prompt_file}' not found")
        return None
    except Exception as e:
        print(f"❌ Error loading prompt: {str(e)}")
        return None

def log_interaction(query, response, strategy, rating=None):
    """Log user interaction to output logs"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "strategy": strategy,
        "rating": rating
    }
    
    # Load existing logs
    log_file = "../evaluation/output_logs.json"
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {
            "evaluation_logs": [],
            "metadata": {
                "model": "llama3:8b",
                "total_queries": 0
            }
        }
    
    # Add new entry
    data["evaluation_logs"].append(log_entry)
    data["metadata"]["total_queries"] = len(data["evaluation_logs"])
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Save back to file
    try:
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"❌ Error saving log: {str(e)}")

def get_user_rating():
    """Get user rating for a response"""
    print("\n📊 Please rate this response (1-5 scale):")
    print("1 = Poor, 2 = Below Average, 3 = Average, 4 = Good, 5 = Excellent")
    
    rating = {}
    
    # Get ratings for each criterion
    criteria = {
        "accuracy": "Mathematical correctness",
        "reasoning_clarity": "Step-by-step explanation quality",
        "hallucinations": "Absence of factual errors (5=no errors, 1=many errors)",
        "consistency": "Appropriate and stable approach"
    }
    
    for criterion, description in criteria.items():
        while True:
            try:
                score = input(f"{criterion.replace('_', ' ').title()} ({description}): ")
                score = int(score)
                if 1 <= score <= 5:
                    rating[criterion] = score
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
    
    # Calculate overall rating
    rating["overall"] = sum(rating.values()) / len(rating)
    rating["timestamp"] = datetime.now().isoformat()
    
    return rating

def display_menu():
    """Display the main menu"""
    print("\n" + "="*50)
    print("🎓 EdTech Math Tutor - Prompt Engineering Lab")
    print("="*50)
    print("1. 💬 Interactive Tutoring Mode")
    print("2. 🔬 Evaluation Mode (Test all strategies)")
    print("3. 📊 Generate Evaluation Report")
    print("4. 🚪 Exit")
    print("="*50)

def analyze_logs():
    """Analyze the logged interactions for patterns"""
    log_file = "../evaluation/output_logs.json"
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        logs = data.get("evaluation_logs", [])
        if not logs:
            print("No logs found for analysis")
            return
        
        # Group by strategy
        strategy_stats = {}
        for log in logs:
            strategy = log.get("strategy", "unknown")
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "ratings": [],
                    "avg_accuracy": 0,
                    "avg_clarity": 0,
                    "avg_hallucinations": 0,
                    "avg_consistency": 0
                }
            
            strategy_stats[strategy]["count"] += 1
            rating = log.get("rating")
            if rating:
                strategy_stats[strategy]["ratings"].append(rating)
        
        # Calculate averages
        for strategy, stats in strategy_stats.items():
            if stats["ratings"]:
                ratings = stats["ratings"]
                stats["avg_accuracy"] = sum(r.get("accuracy", 0) for r in ratings) / len(ratings)
                stats["avg_clarity"] = sum(r.get("reasoning_clarity", 0) for r in ratings) / len(ratings)
                stats["avg_hallucinations"] = sum(r.get("hallucinations", 0) for r in ratings) / len(ratings)
                stats["avg_consistency"] = sum(r.get("consistency", 0) for r in ratings) / len(ratings)
                stats["avg_overall"] = sum(r.get("overall", 0) for r in ratings) / len(ratings)
        
        return strategy_stats
        
    except Exception as e:
        print(f"Error analyzing logs: {str(e)}")
        return None

def generate_report():
    """Generate a summary report of the evaluation"""
    stats = analyze_logs()
    if not stats:
        return
    
    report = []
    report.append("# Evaluation Summary Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## Strategy Performance Summary")
    report.append("| Strategy | Count | Accuracy | Clarity | Hallucinations | Consistency | Overall |")
    report.append("|----------|-------|----------|---------|----------------|-------------|---------|")
    
    for strategy, data in stats.items():
        if data["ratings"]:
            report.append(f"| {strategy} | {data['count']} | {data['avg_accuracy']:.2f} | {data['avg_clarity']:.2f} | {data['avg_hallucinations']:.2f} | {data['avg_consistency']:.2f} | {data['avg_overall']:.2f} |")
    
    report.append("")
    report.append("## Key Findings")
    
    # Find best performing strategy
    best_strategy = max(stats.items(), key=lambda x: x[1].get('avg_overall', 0) if x[1]['ratings'] else 0)
    if best_strategy[1]['ratings']:
        report.append(f"- **Best Overall Strategy**: {best_strategy[0]} (avg: {best_strategy[1]['avg_overall']:.2f})")
    
    # Find strategy with best accuracy
    best_accuracy = max(stats.items(), key=lambda x: x[1].get('avg_accuracy', 0) if x[1]['ratings'] else 0)
    if best_accuracy[1]['ratings']:
        report.append(f"- **Most Accurate Strategy**: {best_accuracy[0]} (avg: {best_accuracy[1]['avg_accuracy']:.2f})")
    
    # Find strategy with best clarity
    best_clarity = max(stats.items(), key=lambda x: x[1].get('avg_clarity', 0) if x[1]['ratings'] else 0)
    if best_clarity[1]['ratings']:
        report.append(f"- **Clearest Explanations**: {best_clarity[0]} (avg: {best_clarity[1]['avg_clarity']:.2f})")
    
    report_text = "\n".join(report)
    
    # Save report
    try:
        with open("../evaluation/summary_report.md", "w") as f:
            f.write(report_text)
        print("✅ Summary report generated: evaluation/summary_report.md")
    except Exception as e:
        print(f"❌ Error saving report: {str(e)}")
    
    return report_text

def validate_setup():
    """Validate that all required files and directories exist"""
    required_files = [
        "../prompts/zero_shot.txt",
        "../prompts/few_shot.txt", 
        "../prompts/cot_prompt.txt",
        "../prompts/meta_prompt.txt",
        "../evaluation/input_queries.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ All required files found")
    return True

def clear_logs():
    """Clear all logged interactions (use with caution)"""
    confirm = input("⚠️  Are you sure you want to clear all logs? (yes/no): ")
    if confirm.lower() == "yes":
        try:
            empty_data = {
                "evaluation_logs": [],
                "metadata": {
                    "model": "llama3:8b",
                    "total_queries": 0,
                    "cleared_on": datetime.now().isoformat()
                }
            }
            with open("../evaluation/output_logs.json", "w") as f:
                json.dump(empty_data, f, indent=2)
            print("✅ Logs cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing logs: {str(e)}")
    else:
        print("❌ Log clearing cancelled")
