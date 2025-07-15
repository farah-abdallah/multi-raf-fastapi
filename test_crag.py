#!/usr/bin/env python3
"""Test script to isolate the gemini-1.0-pro error"""

import tempfile
import os

def test_crag_creation():
    print("Testing CRAG creation...")
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content for CRAG testing.")
        test_file = f.name
    
    try:
        print(f"Created test file: {test_file}")
        
        # Import and create CRAG
        from src.crag import CRAG
        print("CRAG imported successfully")
        
        # Create CRAG instance
        print("Creating CRAG instance...")
        crag = CRAG(test_file)
        print(f"✅ CRAG created successfully with model: {crag.llm.model_name}")
        
        # Test a simple query
        print("Testing simple query...")
        result = crag.run("What is this document about?")
        print(f"✅ Query executed successfully")
        print(f"Result: {result[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"Cleaned up test file: {test_file}")

if __name__ == "__main__":
    test_crag_creation()
