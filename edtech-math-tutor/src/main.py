#!/usr/bin/env python3
"""
EdTech Math Tutor - Main CLI Application
A domain-specific LLM agent for math tutoring (grades 6-10) using different prompt strategies.
"""

import json
import os
import sys
import requests
from datetime import datetime
from utils import load_prompt, log_interaction, get_user_rating, display_menu, validate_setup, generate_report

class MathTutor:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3:8b"
        self.prompt_strategies = ["zero_shot", "few_shot", "cot_prompt", "meta_prompt"]
        self.current_strategy = "zero_shot"
        
    def check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.post(
                "http://localhost:11434/api/show",
                json={"name": self.model},
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_response(self, query, strategy=None):
        """Generate response using specified prompt strategy"""
        if strategy is None:
            strategy = self.current_strategy
            
        # Load the appropriate prompt
        prompt_template = load_prompt(strategy)
        if not prompt_template:
            return f"Error: Could not load prompt strategy '{strategy}'"
        
        # Format the prompt with the user query
        formatted_prompt = prompt_template.replace("{query}", query)
        
        # Make request to Ollama
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: Ollama request failed with status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def interactive_mode(self):
        """Run the tutor in interactive mode"""
        print("🎓 Welcome to the EdTech Math Tutor!")
        print("I'm here to help you with math problems for grades 6-10.")
        print("Type 'help' for commands, 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("📝 Ask me a math question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using the Math Tutor! Keep practicing!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'strategy':
                    self.change_strategy()
                    continue
                elif user_input.lower() == 'menu':
                    display_menu()
                    continue
                elif not user_input:
                    print("Please enter a math question or type 'help' for commands.")
                    continue
                
                # Generate response
                print(f"\n🤖 Using {self.current_strategy} strategy...")
                response = self.generate_response(user_input)
                print(f"\n📚 Math Tutor Response:\n{response}\n")
                
                # Ask for rating (optional)
                rating_choice = input("Would you like to rate this response? (y/n): ").strip().lower()
                if rating_choice == 'y':
                    rating = get_user_rating()
                    log_interaction(user_input, response, self.current_strategy, rating)
                    print("✅ Thank you for your feedback!\n")
                else:
                    log_interaction(user_input, response, self.current_strategy)
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using the Math Tutor!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
    
    def evaluation_mode(self):
        """Run evaluation on test queries"""
        print("🔬 Starting Evaluation Mode...")
        
        # Load test queries
        try:
            with open('../evaluation/input_queries.json', 'r') as f:
                test_data = json.load(f)
            queries = test_data['test_queries']
        except FileNotFoundError:
            print("❌ Error: input_queries.json not found in evaluation folder")
            return
        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON in input_queries.json")
            return
        
        print(f"📊 Found {len(queries)} test queries")
        print("Testing all prompt strategies...\n")
        
        all_results = []
        
        for strategy in self.prompt_strategies:
            print(f"🧪 Testing strategy: {strategy}")
            strategy_results = []
            
            for query_data in queries:
                query = query_data['query']
                print(f"  Query {query_data['id']}: {query[:50]}...")
                
                response = self.generate_response(query, strategy)
                
                # Get manual rating
                print(f"\n📝 Query: {query}")
                print(f"🤖 Response: {response}")
                print(f"✅ Expected: {query_data['expected_answer']}")
                
                rating = get_user_rating()
                
                result = {
                    "query_id": query_data['id'],
                    "query": query,
                    "strategy": strategy,
                    "response": response,
                    "expected_answer": query_data['expected_answer'],
                    "rating": rating,
                    "timestamp": datetime.now().isoformat(),
                    "topic": query_data['topic'],
                    "grade_level": query_data['grade_level']
                }
                
                strategy_results.append(result)
                all_results.append(result)
                print("✅ Logged result\n")
            
            avg_rating = sum(r['rating']['overall'] for r in strategy_results) / len(strategy_results)
            print(f"📈 Average rating for {strategy}: {avg_rating:.2f}\n")
        
        # Save all results
        try:
            with open('../evaluation/output_logs.json', 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {"evaluation_logs": [], "metadata": {}}
        
        existing_data['evaluation_logs'].extend(all_results)
        existing_data['metadata'].update({
            "last_evaluation": datetime.now().isoformat(),
            "total_evaluations": len(all_results)
        })
        
        with open('../evaluation/output_logs.json', 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print("✅ Evaluation complete! Results saved to output_logs.json")
    
    def show_help(self):
        """Display help information"""
        print("\n📖 Math Tutor Commands:")
        print("  help     - Show this help message")
        print("  strategy - Change prompt strategy")
        print("  menu     - Show main menu")
        print("  quit     - Exit the tutor")
        print("\n💡 Tips:")
        print("  - Ask specific math questions")
        print("  - Request step-by-step explanations")
        print("  - Ask for practice problems")
        print("  - Show your work for error correction\n")
    
    def change_strategy(self):
        """Allow user to change prompt strategy"""
        print(f"\n🔧 Current strategy: {self.current_strategy}")
        print("Available strategies:")
        for i, strategy in enumerate(self.prompt_strategies, 1):
            print(f"  {i}. {strategy}")
        
        try:
            choice = input("Select strategy (1-4): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= 4:
                self.current_strategy = self.prompt_strategies[int(choice) - 1]
                print(f"✅ Strategy changed to: {self.current_strategy}\n")
            else:
                print("❌ Invalid choice. Strategy unchanged.\n")
        except ValueError:
            print("❌ Invalid input. Strategy unchanged.\n")

def main():
    """Main function"""
    # Validate setup first
    print("🔍 Validating project setup...")
    if not validate_setup():
        print("❌ Setup validation failed. Please check missing files.")
        sys.exit(1)

    tutor = MathTutor()

    # Check Ollama connection
    if not tutor.check_ollama_connection():
        print("❌ Error: Cannot connect to Ollama or model 'llama3:8b' not found")
        print("Please ensure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Llama 3 8B model is installed (ollama pull llama3:8b)")
        sys.exit(1)

    print("✅ Connected to Ollama with Llama 3 8B model")

    # Show main menu
    display_menu()

    while True:
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            tutor.interactive_mode()
            break
        elif choice == "2":
            tutor.evaluation_mode()
            break
        elif choice == "3":
            print("📊 Generating evaluation report...")
            generate_report()
            break
        elif choice == "4":
            print("👋 Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
