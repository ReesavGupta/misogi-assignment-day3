#!/usr/bin/env python3
"""
Test script to validate the EdTech Math Tutor setup
"""

import os
import sys
import json
import requests

def test_file_structure():
    """Test that all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "README.md",
        "domain_analysis.md", 
        "hallucination_log.md",
        "requirements.txt",
        "setup.py",
        "prompts/zero_shot.txt",
        "prompts/few_shot.txt",
        "prompts/cot_prompt.txt", 
        "prompts/meta_prompt.txt",
        "evaluation/input_queries.json",
        "evaluation/output_logs.json",
        "evaluation/analysis_report.md",
        "src/main.py",
        "src/utils.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_json_files():
    """Test that JSON files are valid"""
    print("🔍 Testing JSON file validity...")
    
    json_files = [
        "evaluation/input_queries.json",
        "evaluation/output_logs.json"
    ]
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            print(f"✅ {file_path} is valid JSON")
        except json.JSONDecodeError as e:
            print(f"❌ {file_path} has invalid JSON: {e}")
            return False
        except FileNotFoundError:
            print(f"❌ {file_path} not found")
            return False
    
    return True

def test_prompt_files():
    """Test that prompt files contain the placeholder"""
    print("🔍 Testing prompt files...")
    
    prompt_files = [
        "prompts/zero_shot.txt",
        "prompts/few_shot.txt", 
        "prompts/cot_prompt.txt",
        "prompts/meta_prompt.txt"
    ]
    
    for file_path in prompt_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if "{query}" not in content:
                print(f"❌ {file_path} missing {{query}} placeholder")
                return False
            else:
                print(f"✅ {file_path} has correct format")
        except FileNotFoundError:
            print(f"❌ {file_path} not found")
            return False
    
    return True

def test_ollama_connection():
    """Test connection to Ollama"""
    print("🔍 Testing Ollama connection...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": "llama3:8b"},
            timeout=5
        )
        if response.status_code == 200:
            print("✅ Ollama connection successful")
            print("✅ Llama 3 8B model available")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def test_python_imports():
    """Test that all required Python modules can be imported"""
    print("🔍 Testing Python imports...")
    
    try:
        sys.path.insert(0, 'src')
        from utils import load_prompt, log_interaction, get_user_rating, display_menu
        print("✅ All utility functions can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_input_queries():
    """Test that input queries are properly formatted"""
    print("🔍 Testing input queries format...")
    
    try:
        with open("evaluation/input_queries.json", 'r') as f:
            data = json.load(f)
        
        queries = data.get("test_queries", [])
        if not queries:
            print("❌ No test queries found")
            return False
        
        required_fields = ["id", "query", "grade_level", "topic", "expected_answer", "difficulty"]
        for query in queries:
            for field in required_fields:
                if field not in query:
                    print(f"❌ Query {query.get('id', 'unknown')} missing field: {field}")
                    return False
        
        print(f"✅ All {len(queries)} test queries properly formatted")
        return True
        
    except Exception as e:
        print(f"❌ Error testing input queries: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 EdTech Math Tutor - Setup Validation")
    print("="*50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("JSON Files", test_json_files),
        ("Prompt Files", test_prompt_files),
        ("Input Queries", test_input_queries),
        ("Python Imports", test_python_imports),
        ("Ollama Connection", test_ollama_connection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("🚀 Run 'cd src && python main.py' to start the tutor")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        print("💡 Try running 'python setup.py' to fix common issues")
    
    print("="*50)

if __name__ == "__main__":
    main()
