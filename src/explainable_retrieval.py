import os
import sys
import io
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from typing import List
import logging

# Import API key manager for rotation
try:
    from api_key_manager import get_api_manager
    API_MANAGER_AVAILABLE = True
    print("🔑 Explainable Retrieval API key rotation manager loaded")
except ImportError:
    API_MANAGER_AVAILABLE = False
    print("⚠️ Explainable Retrieval API key manager not found - using single key mode")

# Set up logging for backend visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path
from src.utils.helpers import *

# Load environment variables from a .env file
load_dotenv()

# Set up API key management
if API_MANAGER_AVAILABLE:
    try:
        api_manager = get_api_manager()
        print(f"🎯 Explainable Retrieval API Manager Status: {api_manager.get_status()}")
    except Exception as e:
        print(f"⚠️ Explainable Retrieval API Manager initialization failed: {e}")
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            print("🔑 Explainable Retrieval using fallback single API key")
        else:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
else:
    # Fallback to single key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        print("🔑 Explainable Retrieval using single API key mode")
    else:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")


class SimpleGeminiLLM:
    """Simple Gemini LLM wrapper with API key rotation support"""
    
    def __init__(self, model_name="gemini-1.5-flash", silent=False):
        self.model_name = model_name
        self.silent = silent
        
        # Try to use API manager if available, otherwise fall back to single key
        global api_manager
        if API_MANAGER_AVAILABLE and 'api_manager' in globals():
            self.use_rotation = True
            if not self.silent:
                print(f"� Explainable Retrieval SimpleGeminiLLM initialized with key rotation for {model_name}")
        else:
            self.use_rotation = False
            # Configure genai with single key
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                if not self.silent:
                    print(f"🔑 Explainable Retrieval SimpleGeminiLLM initialized with single key for {model_name}")
            else:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    def generate_content(self, prompt, max_retries=3):
        """Generate content with automatic key rotation on quota errors"""
        
        if not self.use_rotation:
            # Single key mode
            model = genai.GenerativeModel(self.model_name)
            return model.generate_content(prompt)
        
        # Key rotation mode
        for attempt in range(max_retries):
            try:
                # Get current API key and configure genai
                current_key = api_manager.get_current_key()
                genai.configure(api_key=current_key)
                
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                
                # Successful generation
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for quota/rate limit errors
                if any(err in error_msg for err in ['quota', 'rate limit', '429', 'resource_exhausted']):
                    if not self.silent:
                        print(f"⚠️ Explainable Retrieval Quota/rate limit hit on attempt {attempt + 1}: {e}")
                    
                    if attempt < max_retries - 1:  # Not the last attempt
                        if api_manager.rotate_key():
                            if not self.silent:
                                print(f"🔄 Explainable Retrieval Rotated to new API key, retrying...")
                            continue
                        else:
                            if not self.silent:
                                print("❌ Explainable Retrieval No more API keys available for rotation")
                            raise Exception("All API keys exhausted due to quota limits")
                    else:
                        raise Exception(f"Max retries ({max_retries}) exceeded due to quota limits")
                else:
                    # Non-quota error, don't retry
                    raise e
        
        raise Exception(f"Failed to generate content after {max_retries} attempts")


# Custom embeddings wrapper using sentence-transformers
class SentenceTransformerEmbeddings:
    """Custom embeddings wrapper using sentence-transformers directly"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text])
        return embedding[0].tolist()
    
    def __call__(self, query: str) -> List[float]:
        return self.embed_query(query)


# Define utility classes/functions
class ExplainableRetriever:
    def __init__(self, texts):
        logger.info(f"🔧 Initializing Explainable Retriever with {len(texts)} text chunks")
        self.embeddings = SentenceTransformerEmbeddings()
        logger.info("📊 Creating vector embeddings using SentenceTransformers...")
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        logger.info("✅ Vector store created successfully")
        self.llm = SimpleGeminiLLM('gemini-1.5-flash', silent=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        logger.info("🤖 SimpleGeminiLLM initialized for explanations with key rotation")

    def _generate_explanation(self, query: str, context: str) -> str:
        """Generate explanation using SimpleGeminiLLM with key rotation"""
        logger.info(f"💡 Generating explanation for context chunk (length: {len(context)} chars)")
        prompt = f"""
        Analyze the relationship between the following query and the retrieved context.
        Explain why this context is relevant to the query and how it might help answer the query.

        Query: {query}

        Context: {context}

        Explanation:
        """
        
        try:
            logger.info("🔄 Calling SimpleGeminiLLM for explanation generation...")
            response = self.llm.generate_content(prompt)
            explanation = response.text
            logger.info(f"✅ Explanation generated successfully (length: {len(explanation)} chars)")
            return explanation
        except Exception as e:
            logger.error(f"❌ Error generating explanation: {str(e)}")
            return f"Error generating explanation: {str(e)}"

    def retrieve_and_explain(self, query):
        logger.info(f"🔍 Starting retrieval and explanation for query: '{query}'")
        logger.info("📚 Retrieving relevant documents from vector store...")
        docs = self.retriever.invoke(query)
        logger.info(f"📄 Retrieved {len(docs)} relevant documents")
        
        explained_results = []

        for i, doc in enumerate(docs, 1):
            logger.info(f"📝 Processing document {i}/{len(docs)}")
            logger.info(f"   Content preview: {doc.page_content[:100]}...")
            
            explanation = self._generate_explanation(query, doc.page_content)
            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })
            logger.info(f"   ✅ Explanation generated for document {i}")
        
        logger.info(f"🎯 Completed retrieval and explanation for {len(explained_results)} documents")
        return explained_results


class ExplainableRAGMethod:
    def __init__(self, texts):
        logger.info(f"🚀 Initializing ExplainableRAGMethod with {len(texts)} text chunks")
        self.explainable_retriever = ExplainableRetriever(texts)
        self.llm = SimpleGeminiLLM('gemini-1.5-flash', silent=True)
        logger.info("✅ ExplainableRAGMethod initialization complete")

    def run(self, query):
        logger.info(f"🎯 Running ExplainableRAGMethod for query: '{query}'")
        results = self.explainable_retriever.retrieve_and_explain(query)
        logger.info(f"📊 ExplainableRAGMethod completed - returned {len(results)} explained results")
        return results
    
    def answer(self, query):
        """Generate a comprehensive answer using explained retrievals"""
        logger.info(f"📝 Generating comprehensive answer for query: '{query}'")
        try:
            explained_results = self.explainable_retriever.retrieve_and_explain(query)
            
            if not explained_results:
                logger.warning("⚠️ No relevant documents found for query")
                return "No relevant documents found for your query."
            
            logger.info(f"📚 Combining contexts from {len(explained_results)} explained results...")
            # Combine all content and explanations to generate a comprehensive answer
            combined_context = ""
            explanations = []
            
            for i, result in enumerate(explained_results, 1):
                logger.info(f"   📖 Adding context {i} (length: {len(result['content'])} chars)")
                combined_context += f"\n\nContext {i}: {result['content']}"
                explanations.append(f"Context {i} Relevance: {result['explanation']}")
            
            logger.info("🤖 Generating final comprehensive answer using Gemini...")
            # Generate final answer using Gemini
            answer_prompt = f"""
            Based on the following retrieved contexts and their relevance explanations, provide a comprehensive answer to the user's query.

            Query: {query}

            Retrieved Contexts with Explanations:
            {combined_context}

            Relevance Explanations:
            {' | '.join(explanations)}

            Please provide a well-structured answer that:
            1. Directly addresses the user's query
            2. Synthesizes information from the relevant contexts
            3. Mentions which sources support each part of your answer
            4. Is clear and comprehensive            Answer:            """
            
            response = self.llm.generate_content(answer_prompt)
            
            logger.info(f"✅ Comprehensive answer generated successfully (length: {len(response.text)} chars)")
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error generating comprehensive answer: {str(e)}")
            return f"Error generating comprehensive answer: {str(e)}"
    
    @classmethod
    def from_documents(cls, document_paths: List[str]):
        """Create ExplainableRAGMethod from document file paths"""
        logger.info(f"📁 Creating ExplainableRAGMethod from {len(document_paths)} document paths")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.document_augmentation import load_document_content
        
        all_texts = []
        for i, doc_path in enumerate(document_paths, 1):
            try:
                logger.info(f"📄 Processing document {i}/{len(document_paths)}: {os.path.basename(doc_path)}")
                # Load content from the document using helper function
                content = load_document_content(doc_path)
                doc_name = os.path.basename(doc_path)
                
                logger.info(f"   📝 Document loaded: {len(content)} characters")
                # Split content into chunks for better retrieval
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(content)
                logger.info(f"   🔗 Split into {len(chunks)} chunks")
                  # Add source metadata to each chunk
                for chunk in chunks:
                    all_texts.append(f"[Source: {doc_name}] {chunk}")
                
                logger.info(f"   ✅ Successfully processed {doc_name}")
                    
            except Exception as e:
                logger.error(f"   ❌ Could not process {os.path.basename(doc_path)}: {str(e)}")
                continue
        
        logger.info(f"📚 Created ExplainableRAGMethod with {len(all_texts)} total text chunks")
        return cls(all_texts)

# Argument Parsing
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Explainable RAG Method using Gemini and HuggingFace embeddings",
        epilog="Example: python explainable_retrieval.py --query 'What causes global warming?'"
    )
    parser.add_argument('--query', type=str, default='Why is the sky blue?', 
                       help="Query for the explainable retriever (default: 'Why is the sky blue?')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Sample texts (these can be replaced by actual data)
    texts = [
        "The sky is blue because of the way sunlight interacts with the atmosphere. When sunlight enters Earth's atmosphere, it collides with tiny gas molecules. Blue light waves are shorter and scatter more than other colors, making the sky appear blue.",
        "Photosynthesis is the process by which plants use sunlight to produce energy. Plants absorb carbon dioxide from the air and water from the soil, then use chlorophyll to convert these into glucose and oxygen.",
        "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere. These gases trap heat from the sun, causing the Earth's average temperature to rise over time."
    ]

    print(f"🔍 Processing query: {args.query}")
    print("🔧 Initializing Explainable RAG with Gemini and HuggingFace embeddings...")
    
    try:
        explainable_rag = ExplainableRAGMethod(texts)
        results = explainable_rag.run(args.query)

        print(f"\n📊 Found {len(results)} relevant documents with explanations:\n")
        
        for i, result in enumerate(results, 1):
            print(f"📄 Result {i}:")
            print(f"Content: {result['content']}")
            print(f"💡 Explanation: {result['explanation']}")
            print("-" * 80)
            
    except Exception as e:
        print(f"❌ Error running explainable retrieval: {str(e)}")