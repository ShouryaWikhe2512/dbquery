#!/usr/bin/env python3
"""
Gemini AI Setup Script for Privacy-Preserving RAG Agent
This script helps you set up your Gemini API key for the RAG agent.
"""

import os
import getpass

def setup_gemini():
    """Set up Gemini API key for the RAG agent"""
    print("🔑 Gemini AI Setup for Privacy-Preserving RAG Agent")
    print("=" * 60)
    
    print("\nTo use Gemini AI for intelligent responses, you need an API key.")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Create a new API key")
    print("4. Copy the API key")
    
    print("\n" + "=" * 60)
    
    # Get API key from user
    api_key = getpass.getpass("Enter your Gemini API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("\n⚠️  No API key provided. The RAG agent will use keyword-based responses.")
        print("   You can still use all privacy features, but responses will be basic.")
        return False
    
    # Validate API key format (basic check)
    if len(api_key) < 20:
        print("\n❌ Invalid API key format. Please check your key.")
        return False
    
    # Save to environment file
    env_content = f"""# Gemini AI Configuration
GEMINI_API_KEY={api_key}

# Other environment variables can be added here
# CRIMINAL_FINDER_URL=http://localhost:8000
# PRIVACY_CLIENT_URL=http://localhost:8001
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n✅ Gemini API key saved to .env file")
        print("✅ The RAG agent will now use Gemini AI for intelligent responses!")
        
        # Also set as environment variable for current session
        os.environ['GEMINI_API_KEY'] = api_key
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error saving API key: {e}")
        return False

def test_gemini_connection():
    """Test the Gemini API connection"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ No Gemini API key found in environment")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test
        response = model.generate_content("Hello! Please respond with 'Gemini is working!'")
        
        if "Gemini is working" in response.text:
            print("✅ Gemini API connection successful!")
            return True
        else:
            print("⚠️  Gemini API connected but response format unexpected")
            return False
            
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("   Install it with: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Privacy-Preserving RAG Agent - Gemini AI Setup")
    print("=" * 60)
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("\n📁 .env file already exists")
        choice = input("Do you want to update the Gemini API key? (y/n): ").lower()
        if choice != 'y':
            print("Setup cancelled.")
            return
    
    # Run setup
    if setup_gemini():
        print("\n🧪 Testing Gemini API connection...")
        if test_gemini_connection():
            print("\n🎉 Setup complete! Your RAG agent is ready to use Gemini AI.")
            print("\nTo start the agent:")
            print("python privacy_preserving_rag_agent.py")
        else:
            print("\n⚠️  Setup completed but Gemini API test failed.")
            print("   Check your API key and try again.")
    else:
        print("\n❌ Setup failed. Please try again.")

if __name__ == "__main__":
    main()
