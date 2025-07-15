import os
import time
import random
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    from src.llm.api_manager import get_api_manager
    API_MANAGER_AVAILABLE = True
    print("🔑 CRAG API key rotation manager loaded")
except ImportError:
    API_MANAGER_AVAILABLE = False
    print("⚠️ CRAG API key manager not found - using single key mode")


# Set up API key management
if API_MANAGER_AVAILABLE:
    try:
        api_manager = get_api_manager()
        print(f"🎯 CRAG API Manager Status: {api_manager.get_status()}")
    except Exception as e:
        print(f"⚠️ CRAG API Manager initialization failed: {e}")
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            print("🔑 CRAG using fallback single API key")
else:
    # Fallback to single key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        print("🔑 CRAG using single API key mode")



class SimpleGeminiLLM:
    """Enhanced wrapper for Google Gemini API with Key Rotation"""
    
    def __init__(self, model="gemini-1.5-flash", max_tokens=1000, temperature=0):
        print(f"🔍 DEBUG: SimpleGeminiLLM.__init__ called with model='{model}'")
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set up generation config
        self.generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Set up API manager if available
        if API_MANAGER_AVAILABLE:
            try:
                from src.llm.api_manager import get_api_manager
                self.api_manager = get_api_manager()
                self.use_rotation = True
                print(f"🔑 SimpleGeminiLLM using API rotation with {len(self.api_manager.api_keys)} keys")
            except Exception as e:
                print(f"⚠️ API manager failed, using single key: {e}")
                self.api_manager = None
                self.use_rotation = False
        else:
            self.api_manager = None
            self.use_rotation = False
        
        # Try to get API key from environment if no rotation
        if not self.use_rotation:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
            else:
                raise ValueError("No API key found in environment variables")
        
        # Initialize the model
        self.model = genai.GenerativeModel(model)
    
    def invoke(self, prompt_text, max_retries=3):
        """Invoke the model with automatic key rotation on quota errors"""
        for attempt in range(max_retries):
            try:
                # Add a small delay between requests to avoid rate limits
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying after {delay:.2f} seconds...")
                    time.sleep(delay)
                
                response = self.model.generate_content(
                    prompt_text,
                    generation_config=self.generation_config
                )
                return SimpleResponse(response.text)
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"🔍 DEBUG: Error on attempt {attempt + 1}: {error_msg}")
                
                # Check if it's a quota error and we have API rotation
                if self.use_rotation and self.api_manager and ("quota" in error_msg or "429" in error_msg):
                    if self.api_manager.handle_quota_error(error_msg):
                        # Key switched successfully, reinitialize model with new key
                        self.model = genai.GenerativeModel(self.model_name)
                        print(f"🔄 CRAG retrying with new API key... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        # No more keys available
                        print("❌ All API keys exhausted!")
                        raise Exception(f"All API keys exhausted: {e}")
                
                # Handle rate limit or quota errors without rotation
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    print(f"Rate limit or quota hit on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Quota exceeded: {e}")
                    continue
                
                # Other errors - don't retry
                print(f"Non-quota error on attempt {attempt + 1}: {e}")
                raise Exception(f"Gemini API error: {e}")
    
        # All retries exhausted
        raise Exception(f"CRAG: All API attempts exhausted after {max_retries} attempts")
class SimpleResponse:
    """Simple response wrapper"""
    def __init__(self, content):
        self.content = content


# class SimpleGeminiLLM:
#     """Enhanced wrapper for Google Gemini API with Key Rotation"""
#     def __init__(self, model="gemini-1.5-flash",use_rotation=None, max_tokens=1000, temperature=0, api_manager=None):
#         print(f"🔍 DEBUG: SimpleGeminiLLM.__init__ called with model='{model}'")
#         self.model_name = model
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.api_manager = api_manager
#         self.model = genai.GenerativeModel(model)
#         self.use_rotation=use_rotation

#     def invoke(self, prompt_text, max_retries=3):
#         """
#         Invoke the model with robust retry logic for rate limits and automatic key rotation on quota errors.
#         """
#         import time, random
#         for attempt in range(max_retries):
#             try:
#                 # Add a small delay between requests to avoid rate limits
#                 if attempt > 0:
#                     delay = (2 ** attempt) + random.uniform(0, 1)
#                     print(f"Retrying after {delay:.2f} seconds...")
#                     time.sleep(delay)
#                 response = self.model.generate_content(
#                     prompt_text,
#                     generation_config=genai.types.GenerationConfig(
#                         max_output_tokens=self.max_tokens,
#                         temperature=self.temperature,
#                     )
#                 )
#                 return SimpleResponse(response.text)
#             except Exception as e:
#                 error_msg = str(e).lower()
#                 # Check if it's a quota error and we have API rotation
#                 if self.use_rotation and self.api_manager and self.api_manager.handle_quota_error(error_msg):
#                     self.model = genai.GenerativeModel(self.model_name)
#                     print(f"🔄 CRAG retrying with new API key... (attempt {attempt + 1}/{max_retries})")
#                     continue
#                 # Handle rate limit or quota errors
#                 if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
#                     print(f"Rate limit or quota hit on attempt {attempt + 1}: {e}")
#                     if attempt == max_retries - 1:
#                         raise
#                     continue
#                 # Other errors
#                 print(f"Error on attempt {attempt + 1}: {e}")
#                 if attempt == max_retries - 1:
#                     raise Exception(f"Gemini API error: {e}")
#         # All retries exhausted
#         raise Exception(f"CRAG: All API attempts exhausted after {max_retries} attempts")
# class SimpleResponse:
#     """Simple response wrapper"""
#     def __init__(self, content):
#         self.content = content


