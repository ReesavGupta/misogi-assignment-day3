#!/usr/bin/env python3
"""
Setup script for EdTech Math Tutor
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is 3.7+"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install Python requirements"""
    try:
        print("📦 Installing Python requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Python requirements")
        return False

def check_ollama():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama not found")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found")
        return False

def setup_ollama_model():
    """Setup Llama 3 8B model"""
    try:
        print("🤖 Checking for Llama 3 8B model...")
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "llama3:8b" in result.stdout:
            print("✅ Llama 3 8B model already available")
            return True
        else:
            print("📥 Downloading Llama 3 8B model (this may take a while)...")
            subprocess.check_call(["ollama", "pull", "llama3:8b"])
            print("✅ Llama 3 8B model downloaded")
            return True
    except subprocess.CalledProcessError:
        print("❌ Failed to setup Llama 3 8B model")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up EdTech Math Tutor...")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\n📋 To install Ollama:")
        print("1. Visit: https://ollama.ai/")
        print("2. Download and install Ollama for your OS")
        print("3. Run this setup script again")
        sys.exit(1)
    
    # Setup Ollama model
    if not setup_ollama_model():
        print("\n📋 To manually setup the model:")
        print("1. Run: ollama pull llama3:8b")
        print("2. Wait for download to complete")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🎉 Setup complete!")
    print("="*50)
    print("To start the math tutor:")
    print("1. cd src/")
    print("2. python main.py")
    print("="*50)

if __name__ == "__main__":
    main()
