"""
Google API Key Manager with Rotation and Quota Handling
Automatically switches between multiple API keys when quotas are exceeded
"""

import os
import time
import random
from typing import List, Optional, Dict
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class APIKeyManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.failed_keys = set()
        self.last_switch_time = 0
        self.switch_cooldown = 2  # seconds between switches
        
        if not self.api_keys:
            raise ValueError("No API keys found! Please add GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc. to your .env file")
        
        print(f"🔑 Loaded {len(self.api_keys)} API keys for rotation")
        self._set_current_key()
    
    def _load_api_keys(self) -> List[str]:
        """Load all available API keys from environment variables"""
        keys = []

        # Load from comma-separated GOOGLE_API_KEYS
        keys_env = os.getenv('GOOGLE_API_KEYS')
        if keys_env:
            for key in keys_env.split(','):
                key = key.strip()
                if key and key not in keys:
                    keys.append(key)

        # Try numbered keys (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.)
        for i in range(1, 21):  # Check up to 20 keys
            key = os.getenv(f'GOOGLE_API_KEY_{i}')
            if key and key.strip() and key.strip() not in keys:
                keys.append(key.strip())

        # Try backup keys
        for i in range(1, 11):  # Check up to 10 backup keys
            key = os.getenv(f'GOOGLE_API_KEY_BACKUP_{i}')
            if key and key.strip() and key.strip() not in keys:
                keys.append(key.strip())

        # Try the original key
        original_key = os.getenv('GOOGLE_API_KEY')
        if original_key and original_key.strip() and original_key.strip() not in keys:
            keys.append(original_key.strip())

        return keys
    
    def _set_current_key(self):
        """Set the current API key in the Google AI client"""
        if self.current_key_index < len(self.api_keys):
            current_key = self.api_keys[self.current_key_index]
            genai.configure(api_key=current_key)
            print(f"🔄 Using API key #{self.current_key_index + 1}")
        else:
            raise RuntimeError("All API keys have been exhausted!")
    
    def get_current_key(self) -> str:
        """Get the current API key"""
        if self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        raise RuntimeError("No valid API key available")
    
    def switch_to_next_key(self, mark_current_failed: bool = True):
        """Switch to the next available API key"""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_switch_time < self.switch_cooldown:
            time.sleep(self.switch_cooldown)
        
        if mark_current_failed:
            self.failed_keys.add(self.current_key_index)
            print(f"❌ Marking API key #{self.current_key_index + 1} as failed")
        
        # Find next available key
        original_index = self.current_key_index
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            if self.current_key_index not in self.failed_keys:
                self._set_current_key()
                self.last_switch_time = time.time()
                print(f"✅ Switched to API key #{self.current_key_index + 1}")
                return
        
        # If all keys failed, reset and try again (quotas might have refreshed)
        if len(self.failed_keys) == len(self.api_keys):
            print("⚠️ All keys failed, resetting and retrying...")
            self.failed_keys.clear()
            self.current_key_index = random.randint(0, len(self.api_keys) - 1)
            self._set_current_key()
            self.last_switch_time = time.time()
    
    def handle_quota_error(self, error_message: str) -> bool:
        """
        Handle quota exceeded errors by switching to next key
        Returns True if switched successfully, False if all keys exhausted
        """
        if "quota" in error_message.lower() or "429" in error_message:
            print(f"🚫 Quota exceeded for API key #{self.current_key_index + 1}")
            self.switch_to_next_key(mark_current_failed=True)
            return True
        return False
    
    def get_status(self) -> Dict:
        """Get current status of API key manager"""
        return {
            'total_keys': len(self.api_keys),
            'current_key_index': self.current_key_index + 1,
            'failed_keys': len(self.failed_keys),
            'available_keys': len(self.api_keys) - len(self.failed_keys),
            'current_key_preview': f"{self.get_current_key()[:8]}..." if self.api_keys else "None"
        }
    
    def reset_failed_keys(self):
        """Reset all failed keys (quotas might have refreshed)"""
        print("🔄 Resetting all failed keys - quotas may have refreshed")
        self.failed_keys.clear()
        self.current_key_index = 0
        self._set_current_key()

    def rotate_key(self):
        """Rotate to the next API key (for compatibility with older code)."""
        self.switch_to_next_key()
        return True


# Global instance
_api_key_manager = None

def get_api_manager() -> APIKeyManager:
    """Get the global API key manager instance"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager

def configure_with_rotation():
    """Configure Google AI with key rotation support"""
    manager = get_api_manager()
    return manager


def _call_llm_with_retry(prompt_text, llm, max_retries=3):
    """
    Call LLM with retry logic to handle rate limits.
    """
    for attempt in range(max_retries):
        try:
            # Add a small delay between requests to avoid rate limits
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying after {delay:.2f} seconds...")
                time.sleep(delay)
            
            result = llm.invoke(prompt_text)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                print(f"Rate limit hit on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
            else:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
    
    raise Exception("Max retries exceeded")