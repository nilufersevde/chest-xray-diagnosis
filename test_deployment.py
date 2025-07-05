#!/usr/bin/env python3
"""
Test script to verify deployment readiness
"""
import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import fastapi
        import uvicorn
        import torch
        import torchvision
        import PIL
        import numpy
        import sklearn
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test if the model can be loaded"""
    try:
        from backend.model.load_model import load_trained_model
        model = load_trained_model()
        if model is not None:
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Model is None")
            return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_app_creation():
    """Test if the FastAPI app can be created"""
    try:
        from backend.main import app
        print("✅ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False

def main():
    print("Testing deployment readiness...")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test model loading
    if not test_model_loading():
        success = False
    
    # Test app creation
    if not test_app_creation():
        success = False
    
    print("=" * 40)
    if success:
        print("✅ All tests passed! Deployment should work.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 