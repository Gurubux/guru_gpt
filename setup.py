#!/usr/bin/env python3
"""
Simple setup script for GPT-4o Chat Assistant
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_env_file():
    """Set up environment file"""
    if not os.path.exists(".env"):
        if os.path.exists("env_template.txt"):
            import shutil
            shutil.copy("env_template.txt", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env and add your OpenAI API key")
        else:
            with open(".env", "w") as f:
                f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            print("✅ Created .env file")
            print("⚠️  Please edit .env and add your OpenAI API key")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🤖 GPT-4o Chat Assistant Setup")
    print("=" * 40)
    
    check_python_version()
    install_dependencies()
    setup_env_file()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env and add your OpenAI API key")
    print("2. Run: streamlit run app.py")
    print("3. Open your browser at http://localhost:8501")

if __name__ == "__main__":
    main()
