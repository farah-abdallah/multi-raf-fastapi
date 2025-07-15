import os
from dotenv import load_dotenv
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

# First load the .env file
load_dotenv()

# Check if the variables are loaded
print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')}")
print(f"GOOGLE_API_KEYS: {os.getenv('GOOGLE_API_KEYS')}")

# Parse the GOOGLE_API_KEYS
if os.getenv('GOOGLE_API_KEYS'):
    keys = os.getenv('GOOGLE_API_KEYS').split(',')
    print(f"Found {len(keys)} keys in GOOGLE_API_KEYS:")
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {key[:10]}...")

# Now try to import and use the APIKeyManager
try:
    from src.llm.api_manager import get_api_manager
    
    api_manager = get_api_manager()
    print("\nAPI Manager Status:")
    print(api_manager.get_status())
    
    # Print all keys found by the manager
    print("\nAll keys found by API Manager:")
    if hasattr(api_manager, '_api_keys'):
        for i, key in enumerate(api_manager._api_keys, 1):
            print(f"  {i}. {key[:10]}...")
    else:
        print("  No _api_keys attribute found in API Manager")
        
except Exception as e:
    print(f"Error importing or using APIKeyManager: {e}")