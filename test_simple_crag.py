"""
Simple test script for CRAG functionality
"""

import os
import sys
import tempfile
sys.path.insert(0, os.path.abspath(os.getcwd()))

def test_crag():
    from src.crag import CRAG
    print("Testing CRAG initialization...")
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content for CRAG testing.")
        test_file = f.name
    
    print(f"Created test file: {test_file}")
    
    try:
        # Import and create CRAG instance
        print("CRAG imported successfully")
        print("Creating CRAG instance...")
        
        crag = CRAG(file_path=test_file)
        print(f"✅ CRAG created successfully with model: {crag.llm.model_name}")
        
        # Test a simple query
        print("Testing simple query...")
        query = "What is this document about?"
        response = crag.run(query)
        print("✅ Query executed successfully")
        print(f"Result: {response[:100]}{'...' if len(response) > 100 else ''}")
    
    finally:
        # Clean up the test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up test file: {test_file}")

if __name__ == "__main__":
    test_crag()
