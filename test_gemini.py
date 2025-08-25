#!/usr/bin/env python3
"""
Test Gemini API connection
"""

import google.generativeai as genai

def test_gemini():
    try:
        # Configure with your API key
        api_key = "AIzaSyBH_hrTA13ftEzL3m7Awhhx1svJhKoRPm4"
        genai.configure(api_key=api_key)
        
        # Create model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test simple response
        response = model.generate_content('Hello! Please respond with "Gemini is working!"')
        
        print("✅ Gemini API Test Successful!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API Test Failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini()
