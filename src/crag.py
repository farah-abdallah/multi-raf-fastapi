import os
import sys
import argparse
import time
import random
import io
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchResults
from src.utils.helpers import encode_pdf, encode_document
import json
import requests
from urllib.parse import quote
import streamlit as st

# Import API key manager for rotation
try:
    from api_key_manager import get_api_manager
    API_MANAGER_AVAILABLE = True
    print("🔑 CRAG API key rotation manager loaded")
except ImportError:
    API_MANAGER_AVAILABLE = False
    print("⚠️ CRAG API key manager not found - using single key mode")

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks

# Load environment variables from a .env file
load_dotenv()

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
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
          # Set up API manager if available
        if API_MANAGER_AVAILABLE:
            try:
                self.api_manager = get_api_manager()
                self.use_rotation = True
            except:
                self.api_manager = None
                self.use_rotation = False
        else:
            self.api_manager = None
            self.use_rotation = False
        
        self.model = genai.GenerativeModel(model)
    
    def invoke(self, prompt_text, max_retries=3):
        """Invoke the model with automatic key rotation on quota errors"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                )
                return SimpleResponse(response.text)
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a quota error and we have API rotation
                if self.use_rotation and self.api_manager and self.api_manager.handle_quota_error(error_msg):
                    # Key switched, reinitialize model with new key
                    self.model = genai.GenerativeModel(self.model_name)
                    print(f"🔄 CRAG retrying with new API key... (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    # Non-quota error or no rotation available
                    if attempt < max_retries - 1:
                        print(f"❌ CRAG API Error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                        continue
                    else:
                        raise Exception(f"Gemini API error: {error_msg}")
        
        # All retries exhausted
        raise Exception(f"CRAG: All API attempts exhausted after {max_retries} attempts")
class SimpleResponse:
    """Simple response wrapper"""
    def __init__(self, content):
        self.content = content


class   CRAG:
    """
    A class to handle the CRAG process for document retrieval, evaluation, and knowledge refinement.
    """

    def __init__(self, file_path, model="gemini-1.5-flash", max_tokens=1000, temperature=0, lower_threshold=0.3,
                 upper_threshold=0.7, web_search_enabled=None):
        """
        Initializes the CRAG Retriever by encoding the document and creating the necessary models and search tools.

        Args:
            file_path (str or list): Path(s) to the document file(s) to encode (PDF, TXT, CSV, JSON, DOCX, XLSX).
            model (str): The language model to use for the CRAG process.
            max_tokens (int): Maximum tokens to use in LLM responses (default: 1000).
            temperature (float): The temperature to use for LLM responses (default: 0).
            lower_threshold (float): Lower threshold for document evaluation scores (default: 0.3).
            upper_threshold (float): Upper threshold for document evaluation scores (default: 0.7).
            web_search_enabled (bool or None): If True, enable web search; if False, disable; if None, use env var.
        """
        print("\n--- Initializing CRAG Process ---")
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

        # Allow explicit override, else fallback to environment variable
        if web_search_enabled is not None:
            self.web_search_enabled = bool(web_search_enabled)
        else:
            self.web_search_enabled = os.getenv('CRAG_WEB_SEARCH', 'true').lower() == 'true'

        self.fallback_mode = os.getenv('CRAG_FALLBACK_MODE', 'true').lower() == 'true'

        print(f"Web search enabled: {self.web_search_enabled}")
        print(f"Fallback mode enabled: {self.fallback_mode}")

        # Encode all uploaded documents into a single vector store
        self.vectorstore = encode_document(file_path)
        
        # Store file path for document viewing
        self.file_path = file_path if isinstance(file_path, list) else [file_path]

        # Initialize Gemini language model
        self.llm = SimpleGeminiLLM(model=model, max_tokens=max_tokens, temperature=temperature)

        # Initialize search tool only if web search is enabled
        if self.web_search_enabled:
            self.search = DuckDuckGoSearchResults()
        else:
            self.search = None
            print("Web search disabled via configuration")
        
        # Initialize source tracking
        self._last_source_chunks = []
        self._last_sources = []

    @staticmethod
    def retrieve_documents(query, faiss_index, k=8):
        # Increase k for more robust retrieval (option 1)
        docs = faiss_index.similarity_search(query, k=k)
        return [(doc.page_content, doc.metadata) for doc in docs]

    @staticmethod
    def retrieve_documents_per_document(query, faiss_index, k=3):
        # Option 4: Per-document retrieval (retrieve top k from each document, then combine)
        # Assumes that faiss_index has a method to get all unique sources (documents)
        # and can filter by source. If not, we simulate by grouping after retrieval.
        # Retrieve a large pool, then group by source.
        docs = faiss_index.similarity_search(query, k=30)  # Large pool
        # Group by source
        from collections import defaultdict
        source_to_docs = defaultdict(list)
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown document')
            source_to_docs[source].append(doc)
        # For each document, take top k
        selected = []
        for doclist in source_to_docs.values():
            selected.extend(doclist[:k])
        return [(doc.page_content, doc.metadata) for doc in selected]

    def evaluate_documents(self, query, documents):
        return [self.retrieval_evaluator(query, doc) for doc in documents]

    def retrieval_evaluator(self, query, document):
        prompt_text = f"""You are a strict relevance evaluator. Rate how relevant this document is to the query on a scale from 0 to 1.

IMPORTANT SCORING CRITERIA:
- Score 0.8-1.0: Document DIRECTLY answers the specific question with concrete information
- Score 0.5-0.7: Document contains some relevant information but may be too general
- Score 0.2-0.4: Document mentions the topic but doesn't directly address the query
- Score 0.0-0.1: Document is off-topic or only provides general context

BE EXTRA STRICT WITH:
- Generic introductions, definitions, or background information that don't answer the question
- Content that mentions the topic but doesn't answer the specific question
- Off-topic sections when specific information is requested

SPECIAL CONSIDERATIONS:
- Mental health includes cognitive functions, emotional regulation, mood, anxiety, depression, psychological well-being
- Emotional health relates to mood swings, anxiety, depression, emotional regulation, psychological state
- Physical health relates to immune system, heart health, metabolism, body systems (unless asking about brain/nervous system)
- If asking about mental/emotional health, prioritize content about cognition, emotions, mood, psychology over purely physical health

Query: {query}
Document: {document}

Provide ONLY a number between 0 and 1. Consider: Does this document directly and specifically answer the query?
Relevance score:"""
        
        result = self._call_llm_with_retry(prompt_text)
          # Extract the numeric score from the response
        try:
            score_text = result.content.strip()
            # Try to extract a number from the response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|1|0', score_text)
            if numbers:
                score = float(numbers[0])
                score = min(max(score, 0.0), 1.0)  # Ensure it's between 0 and 1
                
                # Additional filtering based on content analysis
                doc_lower = document.lower()
                query_lower = query.lower()
                
                # More precise penalization - only penalize physical health when specifically asking about mental/cognitive
                if ("cognitive" in query_lower or "mental" in query_lower or "emotional" in query_lower or "psychological" in query_lower):
                    # Only penalize if it's clearly about physical health benefits and NOT about mental/cognitive
                    if ("physical health" in doc_lower or "immune system" in doc_lower or "heart health" in doc_lower or "metabolism" in doc_lower) and not any(term in doc_lower for term in ["cognitive", "mental", "emotional", "psychological", "brain", "memory", "attention", "mood"]):
                        score = score * 0.4  # Moderate penalty for purely physical health content
                        print(f"🔍 CRAG: Penalized physical health chunk when asking about mental/cognitive - reduced score to {score:.2f}")
                
                # Penalize very short generic introductions
                if len(document) < 100 and any(phrase in doc_lower for phrase in ["this document", "we will", "introduction", "overview"]):
                    score = score * 0.4
                    print(f"🔍 CRAG: Penalized short generic intro - reduced score to {score:.2f}")
                
                return score
            else:
                return 0.3  # Lower default score for unparseable responses
        except (ValueError, AttributeError):
            return 0.3  # Lower default score if parsing fails
    
    def knowledge_refinement(self, document):
        prompt_text = f"""Extract ALL the information that is explicitly stated in the provided document.
DO NOT add external knowledge, explanations, or details not present in the text.

STRICT REQUIREMENTS:
- Use ONLY information directly from the document
- Do NOT elaborate or explain beyond what's written
- Do NOT add general knowledge about the topic
- If the document doesn't provide details, don't invent them
- Extract ALL relevant information present in the text, not just parts of it
- Preserve the complete meaning of statements
- Include both positive and negative aspects if mentioned
- Maximum 5 key points per document to ensure completeness

Document:
{document}

Extract ALL the explicit information found in the text (preserve complete statements):"""
        
        result = self._call_llm_with_retry(prompt_text)
        
        # Parse the bullet points from the response
        try:
            content = result.content.strip()
            points = [line.strip() for line in content.split('\n') if line.strip()]
            return points
        except AttributeError:
            return ["Error processing document"]

    def rewrite_query(self, query):
        prompt_text = f"""Rewrite the following query to make it more suitable for a web search:

{query}

Rewritten query:"""
        
        result = self._call_llm_with_retry(prompt_text)
          # Extract the rewritten query from the response
        try:
            return result.content.strip()
        except AttributeError:
            return query  # Return original query if parsing fails

    @staticmethod
    def parse_search_results(results_string):
        try:
            results = json.loads(results_string)
            return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
        except json.JSONDecodeError:
            print("Error parsing search results. Returning empty list.")
            return []

    def perform_web_search(self, query):
        """Improved web search with robust debug logging and user-facing error handling"""
        # Check if web search is disabled
        if not self.web_search_enabled:
            print("Web search disabled via configuration")
            return ["Web search disabled"], [("Configuration", "")]
        try:
            # Try improved web search first
            web_results = self.safe_web_search(query)
            print("\n[CRAG DEBUG] Raw web search results:", web_results)
            if web_results:
                # Process successful results
                web_knowledge = []
                sources = []
                for result in web_results:
                    if isinstance(result, dict):
                        title = result.get('title', 'Untitled')
                        content = result.get('snippet', result.get('content', ''))
                        link = result.get('link', result.get('url', ''))
                        if content:
                            web_knowledge.append(content)
                            # Always label as web search
                            sources.append((f"Web search: {title}", link))
                if web_knowledge:
                    print(f"[CRAG DEBUG] Parsed web knowledge: {web_knowledge}")
                    print(f"[CRAG DEBUG] Web sources: {sources}")
                    return web_knowledge, sources
                else:
                    print("[CRAG DEBUG] No usable content in web results.")
            # Fallback to original method if improved search fails
            if self.search:  # Only if search tool is available
                print("Falling back to original web search...")
                rewritten_query = self.rewrite_query(query)
                web_results = self.search.run(rewritten_query)
                print(f"[CRAG DEBUG] Fallback web search raw results: {web_results}")
                web_knowledge = self.knowledge_refinement(web_results)
                sources = [(f"Web search: {title}", link) for title, link in self.parse_search_results(web_results)]
                if not web_knowledge or all(not w.strip() for w in web_knowledge):
                    print("[CRAG DEBUG] Fallback web search returned no usable knowledge.")
                    return ["Web search failed to return any results relevant to your query."], [("Web Search Failure", "")]
                return web_knowledge, sources
            else:
                print("No search tool available")
                return ["Web search unavailable"], [("No Search Tool", "")]
        except Exception as e:
            print(f"Web search failed: {str(e)}")
            # Return empty results instead of crashing
            return [f"Web search unavailable at this time. Error: {str(e)}"], [("Web Search Error", "")]
    
    def safe_web_search(self, query, max_results=3):
        """Improved web search with proper error handling"""
        try:
            # Clean and encode the query
            clean_query = query.strip().replace('\n', ' ')[:200]  # Limit length
            encoded_query = quote(clean_query)
            
            # Use proper headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Referer': 'https://duckduckgo.com/',
            }
            
            # Try DuckDuckGo with improved parameters
            url = "https://html.duckduckgo.com/html"
            params = {
                'q': clean_query,
                'b': '',
                'kl': 'us-en',  # Change from 'wt-wt' to avoid issues
                'df': 'y'
            }
            
            print(f"Searching for: {clean_query}")
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200 and response.content:
                # Parse results from HTML response
                results = self._parse_duckduckgo_html(response.text)
                if results:
                    print(f"Found {len(results)} web results")
                    return results[:max_results]
                else:
                    print("No results found in response")
                    return []
            else:
                print(f"Web search failed: Status {response.status_code}")
                return []
                
        except requests.RequestException as e:
            print(f"Network error during web search: {e}")
            return []
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def _parse_duckduckgo_html(self, html_content):
        """Parse DuckDuckGo HTML results"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            results = []
            # Look for result containers
            result_divs = soup.find_all('div', class_='result')
            
            for div in result_divs[:5]:  # Limit to first 5 results
                try:
                    # Extract title
                    title_link = div.find('a', class_='result__a')
                    title = title_link.get_text(strip=True) if title_link else "Untitled"
                    link = title_link.get('href', '') if title_link else ''
                    
                    # Extract snippet
                    snippet_elem = div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'link': link
                        })
                except Exception as e:
                    print(f"Error parsing individual result: {e}")
                    continue
            
            return results
            
        except ImportError:
            print("BeautifulSoup not available, falling back to simple parsing")
            # Simple fallback parsing without BeautifulSoup
            return self._simple_parse_html(html_content)
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []
    
    def _simple_parse_html(self, html_content):
        """Simple HTML parsing without BeautifulSoup"""
        try:
            import re
            
            # Simple regex to extract basic information
            title_pattern = r'<a[^>]*class="result__a"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'
            
            titles = re.findall(title_pattern, html_content)
            snippets = re.findall(snippet_pattern, html_content)
            
            results = []
            for i, (title, snippet) in enumerate(zip(titles[:3], snippets[:3])):
                results.append({
                    'title': title.strip(),
                    'snippet': snippet.strip(),
                    'link': ''
                })
            
            return results
            
        except Exception as e:
            print(f"Simple parsing failed: {e}")
            return []
    
    def generate_response(self, query, knowledge, sources):
        sources_text = "\n".join([f"- {title}: {link}" if link else f"- {title}" for title, link in sources])

        # Improved source attribution logic
        def is_doc_source(s):
            # Accept any source that is not web search or error as document
            return (
                ("Retrieved document" in s[0]) or
                ("(fallback" in s[0]) or
                ("(web search failed" in s[0]) or
                ("uploaded document" in s[0]) or
                ("Unknown document" in s[0])
            )
        def is_web_source(s):
            return ("Web search" in s[0])
        def is_error_source(s):
            return any(err in s[0] for err in ["No sources available", "Error", "Web search unavailable", "Configuration"])

        doc_sources = [s for s in sources if is_doc_source(s)]
        web_sources = [s for s in sources if is_web_source(s)]
        only_error_sources = all(is_error_source(s) for s in sources)

        if not self.web_search_enabled:
            if doc_sources:
                source_info = "Based on your uploaded document:"
            elif only_error_sources:
                source_info = "No relevant information found in your uploaded document."
            else:
                source_info = "Based on your uploaded document:"
        else:
            if doc_sources and web_sources:
                source_info = "Based on your uploaded document and web search results:"
            elif doc_sources:
                source_info = "Based on your uploaded document:"
            elif web_sources:
                source_info = "Based on web search results:"
            elif only_error_sources:
                source_info = "No relevant information found in your uploaded document or web search."
            else:
                source_info = "Based on your uploaded document:"

        prompt_text = f"""Answer the question using ONLY the information provided in the knowledge base.
DO NOT add information from your general knowledge or external sources.

STRICT GUIDELINES:
- Use ONLY the provided knowledge - nothing more
- Include ALL relevant information from the sources that answers the question
- If the knowledge is limited, be honest about limitations but provide what is available
- Do NOT elaborate beyond the source material
- Present information completely - don't omit parts of statements
- Keep responses focused but comprehensive based on available sources
- Do NOT invent details or explanations not in the sources

CRITICAL: For any information from uploaded documents, include source reference: [Source: DOCUMENT_NAME, page X, paragraph Y]

Query: {query}

{source_info}
Available Knowledge (use ONLY this information):
{knowledge}

Sources:
{sources_text}

Provide a complete answer using ALL the relevant information provided:

Answer:"""

        result = self._call_llm_with_retry(prompt_text)
        
        # Validate response against sources to prevent hallucination
        response_content = result.content
        
        # Basic validation: check response length vs source material
        if hasattr(self, '_last_source_chunks') and self._last_source_chunks:
            source_text = " ".join([chunk.get('text', '') for chunk in self._last_source_chunks])
            source_words = len(source_text.split())
            response_words = len(response_content.split())
            
            if response_words > source_words * 0.5:
                print(f"⚠️ Warning: Response ({response_words} words) may exceed source material ({source_words} words)")
                print("   Response may contain hallucinated information")
        
        # Format source references with color for better readability
        formatted_response = self._format_sources_with_color(response_content)
        
        return formatted_response

    def run(self, query):
        """CRAG with robust multi-document retrieval and fallback"""
        print(f"\nProcessing query: {query}")

        try:
            # Option 4: Per-document retrieval (retrieve top k from each document, then evaluate all together)
            per_doc_k = 3
            retrieved_docs = self.retrieve_documents_per_document(query, self.vectorstore, k=per_doc_k)
            # Option 1: Increase k for global retrieval as well (for fallback)
            global_k = 8
            global_retrieved_docs = self.retrieve_documents(query, self.vectorstore, k=global_k)

            # Combine and deduplicate (by content+source)
            seen = set()
            all_docs = []
            for doc, meta in retrieved_docs + global_retrieved_docs:
                key = (doc, meta.get('source', ''))
                if key not in seen:
                    all_docs.append((doc, meta))
                    seen.add(key)

            eval_scores = self.evaluate_documents(query, [doc[0] for doc in all_docs])

            print(f"\nRetrieved {len(all_docs)} unique document chunks")
            print(f"Evaluation scores: {eval_scores}")

            sources = []
            # Find all chunks above upper threshold
            relevant_chunks = [(doc, meta, score) for (doc, meta), score in zip(all_docs, eval_scores) if score > self.upper_threshold]
            max_score = max(eval_scores) if eval_scores else 0.0
            max_idx = eval_scores.index(max_score) if eval_scores else -1

            if relevant_chunks:
                print("\nAction: Correct - Using relevant document chunks above threshold")
                # Limit chunks to prevent verbosity - use top 3 most relevant
                relevant_chunks = relevant_chunks[:3]
                print(f"🔍 CRAG: Limited to top {len(relevant_chunks)} chunks to prevent verbosity")
                
                # Process each chunk through knowledge refinement
                refined_knowledge_parts = []
                for doc, metadata, score in relevant_chunks:
                    # Apply knowledge refinement to extract key points
                    refined_points = self.knowledge_refinement(doc)
                    if isinstance(refined_points, list):
                        # Allow more points per chunk to capture complete information
                        refined_knowledge_parts.extend(refined_points[:4])  # Max 4 points per chunk
                    else:
                        refined_knowledge_parts.append(str(refined_points))
                    
                    doc_name = self._create_source_name(metadata)
                    sources.append((doc_name, ""))
                    
                    # Store source chunks for document viewing - use more inclusive threshold for highlighting
                    if score >= 0.5:  # More inclusive threshold to capture relevant mental/emotional health content
                        print(f"📌 CRAG: Storing chunk for highlighting (score: {score:.2f})")
                        print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")
                        self._last_source_chunks.append({
                            'text': doc,
                            'source': metadata.get('source', 'Unknown'),
                            'page': metadata.get('page'),
                            'paragraph': metadata.get('paragraph'),
                            'score': score
                        })
                    else:
                        print(f"🚫 CRAG: Skipping chunk for highlighting (score: {score:.2f} < 0.5)")
                        print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")
                
                # Combine all refined knowledge while removing duplicates
                final_knowledge = "\n".join(list(dict.fromkeys(refined_knowledge_parts)))  # Remove duplicates while preserving order
                print(f"\n🔍 CRAG: Refined knowledge from {len(relevant_chunks)} chunks into {len(refined_knowledge_parts)} key points")
            elif max_score < self.lower_threshold:
                print("\nAction: Incorrect - Performing web search")
                try:
                    final_knowledge, sources = self.perform_web_search(query)
                    if not final_knowledge or all(
                        (not k or 'web search' in k.lower() or 'unavailable' in k.lower() or 'error' in k.lower())
                        for k in final_knowledge):
                        print("\nWeb search failed or returned no relevant results.")
                        # --- FIX: Instead of giving up, always use the best available chunk from uploaded docs, even if below threshold ---
                        if all_docs and len(all_docs) > 0:
                            # Find the best available chunk (even if low score)
                            max_idx = eval_scores.index(max_score) if eval_scores else -1
                            if max_idx != -1:
                                best_doc, metadata = all_docs[max_idx]
                                retrieved_knowledge = self.knowledge_refinement(best_doc)
                                doc_name = self._create_source_name(metadata)
                                final_knowledge = "[No relevant chunk found above threshold or from web search. Using best available chunk from your uploaded documents (low confidence):]\n"
                                if isinstance(retrieved_knowledge, list):
                                    final_knowledge += "\n".join(retrieved_knowledge)
                                else:
                                    final_knowledge += str(retrieved_knowledge)
                                sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
                                
                                # Store fallback source chunk for document viewing - only if reasonably relevant
                                self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context="FALLBACK")
                            else:
                                final_knowledge = "I cannot answer your question. The provided sources indicate that a web search did not return any relevant results, and no relevant information was found in your uploaded documents."
                                sources = [("Web search failure", "")]
                        else:
                            final_knowledge = "I cannot answer your question. The provided sources indicate that a web search did not return any relevant results, and no documents were uploaded."
                            sources = [("Web search failure", "")]
                    if isinstance(final_knowledge, list):
                        final_knowledge = "\n".join(final_knowledge)
                except Exception as web_error:
                    print(f"\nWeb search error: {str(web_error)}")
                    if all_docs and len(all_docs) > 0:
                        # Same fix as above for error case
                        max_idx = eval_scores.index(max_score) if eval_scores else -1
                        if max_idx != -1:
                            best_doc, metadata = all_docs[max_idx]
                            retrieved_knowledge = self.knowledge_refinement(best_doc)
                            doc_name = self._create_source_name(metadata)
                            final_knowledge = "[No relevant chunk found above threshold or from web search. Using best available chunk from your uploaded documents (low confidence):]\n"
                            if isinstance(retrieved_knowledge, list):
                                final_knowledge += "\n".join(retrieved_knowledge)
                            else:
                                final_knowledge += str(retrieved_knowledge)
                            sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
                            
                            # Store fallback source chunk for document viewing - only if reasonably relevant
                            self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context="FALLBACK")
                        else:
                            final_knowledge = f"Web search failed and no relevant information was found in your uploaded documents. Error: {str(web_error)}"
                            sources = [("Error", "")]
                    else:
                        final_knowledge = f"Web search failed and no documents were uploaded. Error: {str(web_error)}"
                        sources = [("Error", "")]
            else:
                # Option 2: Always use the best available chunk if none are above threshold (with warning)
                print("\nAction: No chunk above upper threshold, using best available chunk with low-confidence warning")
                if max_idx == -1:
                    final_knowledge = "No relevant information found."
                    sources = [("No sources available", "")]
                else:
                    best_doc, metadata = all_docs[max_idx]
                    retrieved_knowledge = self.knowledge_refinement(best_doc)
                    doc_name = self._create_source_name(metadata)
                    final_knowledge = "[Low confidence: No chunk was highly relevant, but this is the best available information:]\n" + ("\n".join(retrieved_knowledge) if isinstance(retrieved_knowledge, list) else str(retrieved_knowledge))
                    sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
                    
                    # Store fallback source chunk for document viewing - only if reasonably relevant
                    self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context="FALLBACK")

            print("\nFinal knowledge:")
            print(final_knowledge[:500] + "..." if len(str(final_knowledge)) > 500 else str(final_knowledge))

            print("\nSources:")
            for title, link in sources:
                print(f"{title}: {link}" if link else title)

            # Store sources for document viewing
            self._last_sources = sources

            print("\nGenerating response...")
            try:
                response = self.generate_response(query, final_knowledge, sources)
                print("\nResponse generated successfully")
            except Exception as response_error:
                print(f"\nError generating response: {str(response_error)}")
                response = f"Error generating response: {str(response_error)}"

            time.sleep(0.5)
            return response

        except Exception as e:
            print(f"\nCRAG process failed: {str(e)}")
            return f"CRAG process failed: {str(e)}"

    def run_with_sources(self, query):
        """
        Enhanced CRAG that returns answer with detailed source chunks for document viewing
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing:
            - answer: The generated response
            - source_chunks: List of source chunks with metadata
            - sources: List of source information
        """
        print(f"\nProcessing query with source tracking: {query}")

        try:
            # Clear previous source tracking
            self._last_source_chunks = []
            self._last_sources = []
            
            # Run the main CRAG process
            response = self.run(query)
            
            print(f"\n🎯 CRAG: Final source chunks to return for highlighting:")
            print(f"   Total chunks: {len(self._last_source_chunks)}")
            for i, chunk in enumerate(self._last_source_chunks):
                score = chunk.get('score', 'N/A')
                text = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                print(f"   {i+1}. Score: {score}, Text: '{text}'")
            
            return {
                'answer': response,
                'source_chunks': self._last_source_chunks,
                'sources': self._last_sources
            }
            
        except Exception as e:
            print(f"\nCRAG process with sources failed: {str(e)}")
            return {
                'answer': f"CRAG process failed: {str(e)}",
                'source_chunks': [],
                'sources': []
            }

    def _call_llm_with_retry(self, prompt_text, max_retries=3):
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
                
                result = self.llm.invoke(prompt_text)
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

    def _create_source_name(self, metadata):
        """Helper method to create consistent source names with page/paragraph info"""
        raw_source = metadata.get('source', 'Unknown document')
        
        # Extract clean filename from path (handles both Windows and Unix paths)
        import os
        if raw_source and raw_source != 'Unknown document':
            doc_name = os.path.basename(raw_source)
            # Remove any temporary file prefixes if present
            if doc_name.startswith('tmp') and '\\' in doc_name:
                doc_name = doc_name.split('\\')[-1]
        else:
            doc_name = 'Unknown document'
            
        if 'page' in metadata and metadata['page'] not in [None, '', 'None']:
            doc_name += f", page {metadata['page']}"
        if 'paragraph' in metadata and metadata['paragraph'] not in [None, '', 'None']:
            doc_name += f", paragraph {metadata['paragraph']}"
        return doc_name

    def _store_chunk_for_highlighting(self, doc, metadata, score, threshold=0.4, context=""):
        """Helper method to conditionally store chunks for document highlighting"""
        if score >= threshold:
            print(f"📌 CRAG {context}: Storing chunk for highlighting (score: {score:.2f})")
            print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")
            self._last_source_chunks.append({
                'text': doc,
                'source': metadata.get('source', 'Unknown'),
                'page': metadata.get('page'),
                'paragraph': metadata.get('paragraph'),
                'score': score
            })
        else:
            print(f"🚫 CRAG {context}: Skipping chunk for highlighting (score: {score:.2f} < {threshold})")
            print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")

    def _create_fallback_response(self, all_docs, eval_scores, context="FALLBACK"):
        """Helper method to create fallback response from best available chunk"""
        if not all_docs or not eval_scores:
            return "No relevant information found.", [("No sources available", "")]
        
        max_score = max(eval_scores)
        max_idx = eval_scores.index(max_score)
        best_doc, metadata = all_docs[max_idx]
        
        retrieved_knowledge = self.knowledge_refinement(best_doc)
        doc_name = self._create_source_name(metadata)
        
        final_knowledge = "[No relevant chunk found above threshold or from web search. Using best available chunk from your uploaded documents (low confidence):]\n"
        if isinstance(retrieved_knowledge, list):
            final_knowledge += "\n".join(retrieved_knowledge)
        else:
            final_knowledge += str(retrieved_knowledge)
        
        sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
        
        # Store fallback source chunk for document viewing - only if reasonably relevant
        self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context=context)
        
        return final_knowledge, sources

    def _handle_web_search_failure(self, all_docs, eval_scores, error_msg=""):
        """Helper method to handle web search failures with document fallback"""
        if all_docs and len(all_docs) > 0:
            final_knowledge, sources = self._create_fallback_response(
                all_docs, eval_scores, "FALLBACK (web search failed)"
            )
        else:
            error_suffix = f" Error: {error_msg}" if error_msg else ""
            final_knowledge = f"I cannot answer your question. The provided sources indicate that a web search did not return any relevant results, and no relevant information was found in your uploaded documents.{error_suffix}"
            sources = [("Web search failure", "")]
        
        return final_knowledge, sources

    def _format_sources_with_color(self, response_text):
        """
        Post-process the response to add colored formatting to source references
        """
        import re
        
        # Pattern to match source references in brackets: [Source: filename, page X, paragraph Y]
        source_pattern = r'\[Source: ([^\]]+)\]'
        
        def replace_source(match):
            source_text = match.group(1)
            # Format with HTML and CSS class for colored display in Streamlit
            return f'<span class="source-reference">[Source: {source_text}]</span>'
        
        # Replace all source references with colored versions
        formatted_response = re.sub(source_pattern, replace_source, response_text)
        
        return formatted_response

# Function to validate command line inputs
def validate_args(args):
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    if args.temperature < 0 or args.temperature > 1:
        raise ValueError("temperature must be between 0 and 1.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="CRAG Process for Document Retrieval and Query Answering.")
    parser.add_argument("--file_path", type=str, default="data/Understanding_Climate_Change (1).pdf",
                        help="Path to the document file to encode (supports PDF, TXT, CSV, JSON, DOCX, XLSX).")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash",
                        help="Language model to use (default: gemini-1.5-flash).")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum tokens to use in LLM responses (default: 1000).")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature to use for LLM responses (default: 0).")
    parser.add_argument("--query", type=str, default="What are the main causes of climate change?",
                        help="Query to test the CRAG process.")
    parser.add_argument("--lower_threshold", type=float, default=0.3,
                        help="Lower threshold for score evaluation (default: 0.3).")
    parser.add_argument("--upper_threshold", type=float, default=0.7,
                        help="Upper threshold for score evaluation (default: 0.7).")
    parser.add_argument("--web_search_enabled", type=str, default=None,
                        help="Set to 'true' to enable web search, 'false' to disable, or leave unset to use env var.")

    args = parser.parse_args()
    # Convert web_search_enabled to bool or None
    if args.web_search_enabled is not None:
        val = args.web_search_enabled.strip().lower()
        if val == 'true':
            args.web_search_enabled = True
        elif val == 'false':
            args.web_search_enabled = False
        else:
            args.web_search_enabled = None
    return validate_args(args)


# Main function to handle argument parsing and call the CRAG class
def main(args):
    # Initialize the CRAG process
    crag = CRAG(
        file_path=args.file_path,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold,
        web_search_enabled=args.web_search_enabled
    )

    # Process the query
    response = crag.run(args.query)
    print(f"Query: {args.query}")
    print(f"Answer: {response}")



# --- Streamlit UI for CRAG Web Search Toggle ---

def streamlit_crag_app():
    import tempfile
    import os
    st.title("CRAG Chatbot Demo")
    st.write("Upload one or more documents and ask questions. Optionally enable/disable web search.")

    # Allow multiple file uploads
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, CSV, DOCX, XLSX, JSON)", 
        accept_multiple_files=True
    )
    web_search_enabled = st.checkbox("Enable Web Search", value=True)
    query = st.text_input("Enter your question:")

    model = st.selectbox("Model", ["gemini-1.5-flash"], index=0)
    max_tokens = st.number_input("Max tokens", min_value=100, max_value=4096, value=1000)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)

    if uploaded_files and query:
        # Save all uploaded files to temp and collect their paths
        tmp_paths = []
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_paths.append(tmp_file.name)

        st.info(f"Processing {len(tmp_paths)} document(s): " + ", ".join([os.path.basename(p) for p in tmp_paths]))
        # Pass the list of file paths to CRAG
        crag = CRAG(
            file_path=tmp_paths,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            web_search_enabled=web_search_enabled
        )
        with st.spinner("Generating answer..."):
            answer = crag.run(query)
        st.success("Answer:")
        st.write(answer)


# If running as a script, launch Streamlit UI if desired
if __name__ == '__main__':
    import sys
    if any(arg in sys.argv for arg in ['--ui', '--streamlit']):
        streamlit_crag_app()
    else:
        main(parse_args())




# import os
# import time
# import random
# import re

# from src.llm.gemini import SimpleGeminiLLM
# from src.utils.helpers import encode_document
# from src.core.retrieval import retrieve_documents, retrieve_documents_per_document
# from src.core.knowledge import knowledge_refinement, generate_response
# import argparse
# from langchain_community.tools import DuckDuckGoSearchResults 
# from src.core.evaluation import evaluate_documents
# # If you have a modular LLM wrapper, import it here
# # from src.llm.simple_gemini import SimpleGeminiLLM

# class   CRAG:
#     """
#     A class to handle the CRAG process for document retrieval, evaluation, and knowledge refinement.
#     """

#     def __init__(self, file_path, model="gemini-1.5-flash", max_tokens=1000, temperature=0, lower_threshold=0.3,
#                  upper_threshold=0.7, web_search_enabled=None):
#         """
#         Initializes the CRAG Retriever by encoding the document and creating the necessary models and search tools.

#         Args:
#             file_path (str or list): Path(s) to the document file(s) to encode (PDF, TXT, CSV, JSON, DOCX, XLSX).
#             model (str): The language model to use for the CRAG process.
#             max_tokens (int): Maximum tokens to use in LLM responses (default: 1000).
#             temperature (float): The temperature to use for LLM responses (default: 0).
#             lower_threshold (float): Lower threshold for document evaluation scores (default: 0.3).
#             upper_threshold (float): Upper threshold for document evaluation scores (default: 0.7).
#             web_search_enabled (bool or None): If True, enable web search; if False, disable; if None, use env var.
#         """
#         print("\n--- Initializing CRAG Process ---")
#         self.lower_threshold = lower_threshold
#         self.upper_threshold = upper_threshold

#         # Allow explicit override, else fallback to environment variable
#         if web_search_enabled is not None:
#             self.web_search_enabled = bool(web_search_enabled)
#         else:
#             self.web_search_enabled = os.getenv('CRAG_WEB_SEARCH', 'true').lower() == 'true'

#         self.fallback_mode = os.getenv('CRAG_FALLBACK_MODE', 'true').lower() == 'true'

#         print(f"Web search enabled: {self.web_search_enabled}")
#         print(f"Fallback mode enabled: {self.fallback_mode}")

#         # Encode all uploaded documents into a single vector store
#         self.vectorstore = encode_document(file_path)
        
#         # Store file path for document viewing
#         self.file_path = file_path if isinstance(file_path, list) else [file_path]

#         # Initialize Gemini language model with error handling
#         try:
#             self.llm = SimpleGeminiLLM(model=model, max_tokens=max_tokens, temperature=temperature)
#             print(f"✅ LLM initialized successfully with model: {model}")
#         except Exception as e:
#             print(f"❌ Error initializing LLM: {str(e)}")
#             raise Exception(f"Failed to initialize SimpleGeminiLLM: {str(e)}")

#         # Initialize search tool only if web search is enabled
#         if self.web_search_enabled:
#             self.search = DuckDuckGoSearchResults()
#         else:
#             self.search = None
#             print("Web search disabled via configuration")
        
#         # Initialize source tracking
#         self._last_source_chunks = []
#         self._last_sources = []
       

#     @property
#     def source_chunks(self):
#         """Return the source chunks from the last query for document viewer integration"""
#         return self._last_source_chunks

#     def run(self, query):
#         """CRAG with robust multi-document retrieval and fallback"""
#         print(f"\nProcessing query: {query}")

#         try:
#             # Option 4: Per-document retrieval (retrieve top k from each document, then evaluate all together)
#             per_doc_k = 3
#             retrieved_docs = retrieve_documents_per_document(query, self.vectorstore, k=per_doc_k)
#             # Option 1: Increase k for global retrieval as well (for fallback)
#             global_k = 8
#             global_retrieved_docs = retrieve_documents(query, self.vectorstore, k=global_k)

#             # Combine and deduplicate (by content+source)
#             seen = set()
#             all_docs = []
#             for doc, meta in retrieved_docs + global_retrieved_docs:
#                 key = (doc, meta.get('source', ''))
#                 if key not in seen:
#                     all_docs.append((doc, meta))
#                     seen.add(key)

#             eval_scores = evaluate_documents(query, [doc[0] for doc in all_docs])

#             print(f"\nRetrieved {len(all_docs)} unique document chunks")
#             print(f"Evaluation scores: {eval_scores}")

#             sources = []
#             # Find all chunks above upper threshold
#             relevant_chunks = [(doc, meta, score) for (doc, meta), score in zip(all_docs, eval_scores) if score > self.upper_threshold]
#             max_score = max(eval_scores) if eval_scores else 0.0
#             max_idx = eval_scores.index(max_score) if eval_scores else -1

#             if relevant_chunks:
#                 print("\nAction: Correct - Using relevant document chunks above threshold")
#                 # Limit chunks to prevent verbosity - use top 3 most relevant
#                 relevant_chunks = relevant_chunks[:3]
#                 print(f"🔍 CRAG: Limited to top {len(relevant_chunks)} chunks to prevent verbosity")
                
#                 # Process each chunk through knowledge refinement
#                 refined_knowledge_parts = []
#                 for doc, metadata, score in relevant_chunks:
#                     # Apply knowledge refinement to extract key points
#                     refined_points = knowledge_refinement(doc, self.llm)
#                     if isinstance(refined_points, list):
#                         # Allow more points per chunk to capture complete information
#                         refined_knowledge_parts.extend(refined_points[:4])  # Max 4 points per chunk
#                     else:
#                         refined_knowledge_parts.append(str(refined_points))
                    
#                     doc_name = self._create_source_name(metadata)
#                     sources.append((doc_name, ""))
                    
#                     # Store source chunks for document viewing - use more inclusive threshold for highlighting
#                     if score >= 0.5:  # More inclusive threshold to capture relevant mental/emotional health content
#                         print(f"📌 CRAG: Storing chunk for highlighting (score: {score:.2f})")
#                         print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")
#                         self._last_source_chunks.append({
#                             'text': doc,
#                             'source': metadata.get('source', 'Unknown'),
#                             'page': metadata.get('page'),
#                             'paragraph': metadata.get('paragraph'),
#                             'score': score
#                         })
#                     else:
#                         print(f"🚫 CRAG: Skipping chunk for highlighting (score: {score:.2f} < 0.5)")
#                         print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")
                
#                 # Combine all refined knowledge while removing duplicates
#                 final_knowledge = "\n".join(list(dict.fromkeys(refined_knowledge_parts)))  # Remove duplicates while preserving order
#                 print(f"\n🔍 CRAG: Refined knowledge from {len(relevant_chunks)} chunks into {len(refined_knowledge_parts)} key points")
#             elif max_score < self.lower_threshold:
#                 print("\nAction: Incorrect - Performing web search")
#                 try:
#                     final_knowledge, sources = self.perform_web_search(query)
#                     if not final_knowledge or all(
#                         (not k or 'web search' in k.lower() or 'unavailable' in k.lower() or 'error' in k.lower())
#                         for k in final_knowledge):
#                         print("\nWeb search failed or returned no relevant results.")
#                         # --- FIX: Instead of giving up, always use the best available chunk from uploaded docs, even if below threshold ---
#                         if all_docs and len(all_docs) > 0:
#                             # Find the best available chunk (even if low score)
#                             max_idx = eval_scores.index(max_score) if eval_scores else -1
#                             if max_idx != -1:
#                                 best_doc, metadata = all_docs[max_idx]
#                                 retrieved_knowledge = knowledge_refinement(best_doc, self.llm)
#                                 doc_name = self._create_source_name(metadata)
#                                 final_knowledge = "[No relevant chunk found above threshold or from web search. Using best available chunk from your uploaded documents (low confidence):]\n"
#                                 if isinstance(retrieved_knowledge, list):
#                                     final_knowledge += "\n".join(retrieved_knowledge)
#                                 else:
#                                     final_knowledge += str(retrieved_knowledge)
#                                 sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
                                
#                                 # Store fallback source chunk for document viewing - only if reasonably relevant
#                                 self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context="FALLBACK")
#                             else:
#                                 final_knowledge = "I cannot answer your question. The provided sources indicate that a web search did not return any relevant results, and no relevant information was found in your uploaded documents."
#                                 sources = [("Web search failure", "")]
#                         else:
#                             final_knowledge = "I cannot answer your question. The provided sources indicate that a web search did not return any relevant results, and no documents were uploaded."
#                             sources = [("Web search failure", "")]
#                     if isinstance(final_knowledge, list):
#                         final_knowledge = "\n".join(final_knowledge)
#                 except Exception as web_error:
#                     print(f"\nWeb search error: {str(web_error)}")
#                     if all_docs and len(all_docs) > 0:
#                         # Same fix as above for error case
#                         max_idx = eval_scores.index(max_score) if eval_scores else -1
#                         if max_idx != -1:
#                             best_doc, metadata = all_docs[max_idx]
#                             retrieved_knowledge = knowledge_refinement(best_doc, self.llm)
#                             doc_name = self._create_source_name(metadata)
#                             final_knowledge = "[No relevant chunk found above threshold or from web search. Using best available chunk from your uploaded documents (low confidence):]\n"
#                             if isinstance(retrieved_knowledge, list):
#                                 final_knowledge += "\n".join(retrieved_knowledge)
#                             else:
#                                 final_knowledge += str(retrieved_knowledge)
#                             sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
                            
#                             # Store fallback source chunk for document viewing - only if reasonably relevant
#                             self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context="FALLBACK")
#                         else:
#                             final_knowledge = f"Web search failed and no relevant information was found in your uploaded documents. Error: {str(web_error)}"
#                             sources = [("Error", "")]
#                     else:
#                         final_knowledge = f"Web search failed and no documents were uploaded. Error: {str(web_error)}"
#                         sources = [("Error", "")]
#             else:
#                 # Option 2: Always use the best available chunk if none are above threshold (with warning)
#                 print("\nAction: No chunk above upper threshold, using best available chunk with low-confidence warning")
#                 if max_idx == -1:
#                     final_knowledge = "No relevant information found."
#                     sources = [("No sources available", "")]
#                 else:
#                     best_doc, metadata = all_docs[max_idx]
#                     retrieved_knowledge = knowledge_refinement(best_doc, self.llm)
#                     doc_name = self._create_source_name(metadata)
#                     final_knowledge = "[Low confidence: No chunk was highly relevant, but this is the best available information:]\n" + ("\n".join(retrieved_knowledge) if isinstance(retrieved_knowledge, list) else str(retrieved_knowledge))
#                     sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
                    
#                     # Store fallback source chunk for document viewing - only if reasonably relevant
#                     self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context="FALLBACK")

#             print("\nFinal knowledge:")
#             print(final_knowledge[:500] + "..." if len(str(final_knowledge)) > 500 else str(final_knowledge))

#             print("\nSources:")
#             for title, link in sources:
#                 print(f"{title}: {link}" if link else title)

#             # Store sources for document viewing
#             self._last_sources = sources

#             print("\nGenerating response...")
#             try:
#                 response = generate_response(
#                             query,
#                             final_knowledge,
#                             sources,
#                             self.llm,
#                             self._last_source_chunks,
#                             self.web_search_enabled
#                         )
#                 print("\nResponse generated successfully")
#             except Exception as response_error:
#                 print(f"\nError generating response: {str(response_error)}")
#                 response = f"Error generating response: {str(response_error)}"

#             time.sleep(0.5)
#             return response

#         except Exception as e:
#             print(f"\nCRAG process failed: {str(e)}")
#             return f"CRAG process failed: {str(e)}"

#     def run_with_sources(self, query):
#         """
#         Enhanced CRAG that returns answer with detailed source chunks for document viewing
        
#         Args:
#             query: The user's question
            
#         Returns:
#             Dictionary containing:
#             - answer: The generated response
#             - source_chunks: List of source chunks with metadata
#             - sources: List of source information
#         """
#         print(f"\nProcessing query with source tracking: {query}")

#         try:
#             # Clear previous source tracking
#             self._last_source_chunks = []
#             self._last_sources = []
            
#             # Run the main CRAG process
#             response = self.run(query)
            
#             print(f"\n🎯 CRAG: Final source chunks to return for highlighting:")
#             print(f"   Total chunks: {len(self._last_source_chunks)}")
#             for i, chunk in enumerate(self._last_source_chunks):
#                 score = chunk.get('score', 'N/A')
#                 text = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
#                 print(f"   {i+1}. Score: {score}, Text: '{text}'")
            
#             return {
#                 'answer': response,
#                 'source_chunks': self._last_source_chunks,
#                 'sources': self._last_sources
#             }
            
#         except Exception as e:
#             print(f"\nCRAG process with sources failed: {str(e)}")
#             return {
#                 'answer': f"CRAG process failed: {str(e)}",
#                 'source_chunks': [],
#                 'sources': []
#             }

#     # def _process_chunks_in_parallel(self, chunks):
#     #     """Process chunks in parallel to speed up LLM calls"""
#     #     import concurrent.futures
        
#     #     results = []
#     #     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
#     #         # Map function to arguments
#     #         future_to_chunk = {
#     #             executor.submit(knowledge_refinement, doc, self.llm): 
#     #             (doc, metadata, score) for doc, metadata, score in chunks
#     #         }
            
#     #         # Process results as they complete
#     #         for future in concurrent.futures.as_completed(future_to_chunk):
#     #             doc, metadata, score = future_to_chunk[future]
#     #             try:
#     #                 refined_points = future.result()
#     #                 results.append((refined_points, doc, metadata, score))
#     #             except Exception as e:
#     #                 print(f"Error processing chunk: {e}")
#     #                 # Add fallback for failed chunks
#     #                 results.append((["Error processing chunk"], doc, metadata, score))
                    
#     #     return results
            
#     def _create_source_name(self, metadata):
#         """Helper method to create consistent source names with page/paragraph info"""
#         raw_source = metadata.get('source', 'Unknown document')
        
#         # Extract clean filename from path (handles both Windows and Unix paths)
#         import os
#         if raw_source and raw_source != 'Unknown document':
#             doc_name = os.path.basename(raw_source)
#             # Remove any temporary file prefixes if present
#             if doc_name.startswith('tmp') and '\\' in doc_name:
#                 doc_name = doc_name.split('\\')[-1]
#         else:
#             doc_name = 'Unknown document'
            
#         if 'page' in metadata and metadata['page'] not in [None, '', 'None']:
#             doc_name += f", page {metadata['page']}"
#         if 'paragraph' in metadata and metadata['paragraph'] not in [None, '', 'None']:
#             doc_name += f", paragraph {metadata['paragraph']}"
#         return doc_name

#     def _store_chunk_for_highlighting(self, doc, metadata, score, threshold=0.4, context=""):
#         """Helper method to conditionally store chunks for document highlighting"""
#         if score >= threshold:
#             print(f"📌 CRAG {context}: Storing chunk for highlighting (score: {score:.2f})")
#             print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")
#             self._last_source_chunks.append({
#                 'text': doc,
#                 'source': metadata.get('source', 'Unknown'),
#                 'page': metadata.get('page'),
#                 'paragraph': metadata.get('paragraph'),
#                 'score': score
#             })
#         else:
#             print(f"🚫 CRAG {context}: Skipping chunk for highlighting (score: {score:.2f} < {threshold})")
#             print(f"   Text preview: '{doc[:100]}{'...' if len(doc) > 100 else ''}'")

#     def _create_source_info(self):
#         """Helper method to create source info string for response generation"""
#         if not self._last_source_chunks:
#             return "No specific source information available."
        
#         source_info_parts = []
#         for i, chunk in enumerate(self._last_source_chunks, 1):
#             source_name = chunk.get('source', 'Unknown')
#             page = chunk.get('page')
#             paragraph = chunk.get('paragraph')
            
#             # Create source reference
#             source_ref = f"Source {i}: {source_name}"
#             if page and page not in [None, '', 'None']:
#                 source_ref += f", page {page}"
#             if paragraph and paragraph not in [None, '', 'None']:
#                 source_ref += f", paragraph {paragraph}"
            
#             source_info_parts.append(source_ref)
        
#         return "Source references:\n" + "\n".join(source_info_parts)

#     def _create_fallback_response(self, all_docs, eval_scores, context="FALLBACK"):
#         """Helper method to create fallback response from best available chunk"""
#         if not all_docs or not eval_scores:
#             return "No relevant information found.", [("No sources available", "")]
        
#         max_score = max(eval_scores)
#         max_idx = eval_scores.index(max_score)
#         best_doc, metadata = all_docs[max_idx]
        
#         retrieved_knowledge = knowledge_refinement(best_doc, self.llm)
#         doc_name = self._create_source_name(metadata)
        
#         final_knowledge = "[No relevant chunk found above threshold or from web search. Using best available chunk from your uploaded documents (low confidence):]\n"
#         if isinstance(retrieved_knowledge, list):
#             final_knowledge += "\n".join(retrieved_knowledge)
#         else:
#             final_knowledge += str(retrieved_knowledge)
        
#         sources = [(doc_name + " (fallback: best available chunk, low confidence)", "")]
        
#         # Store fallback source chunk for document viewing - only if reasonably relevant
#         self._store_chunk_for_highlighting(best_doc, metadata, max_score, threshold=0.4, context=context)
        
#         return final_knowledge, sources

#     def _handle_web_search_failure(self, all_docs, eval_scores, error_msg=""):
#         """Helper method to handle web search failures with document fallback"""
#         if all_docs and len(all_docs) > 0:
#             final_knowledge, sources = self._create_fallback_response(
#                 all_docs, eval_scores, "FALLBACK (web search failed)"
#             )
#         else:
#             error_suffix = f" Error: {error_msg}" if error_msg else ""
#             final_knowledge = f"I cannot answer your question. The provided sources indicate that a web search did not return any relevant results, and no relevant information was found in your uploaded documents.{error_suffix}"
#             sources = [("Web search failure", "")]
        
#         return final_knowledge, sources

    

#     def perform_web_search(self, query):
#         """Perform web search using DuckDuckGo"""
#         if not self.web_search_enabled or not self.search:
#             return "Web search is disabled", [("Web search disabled", "")]
        
#         try:
#             from src.search.web import perform_web_search
#             return perform_web_search(query)
#         except Exception as e:
#             print(f"Web search error: {e}")
#             return f"Web search failed: {e}", [("Web search error", "")]
    
# #     def generate_response(self, query, knowledge, sources):
# #         sources_text = "\n".join([f"- {title}: {link}" if link else f"- {title}" for title, link in sources])

# #         # Improved source attribution logic
# #         def is_doc_source(s):
# #             # Accept any source that is not web search or error as document
# #             return (
# #                 ("Retrieved document" in s[0]) or
# #                 ("(fallback" in s[0]) or
# #                 ("(web search failed" in s[0]) or
# #                 ("uploaded document" in s[0]) or
# #                 ("Unknown document" in s[0])
# #             )
# #         def is_web_source(s):
# #             return ("Web search" in s[0])
# #         def is_error_source(s):
# #             return any(err in s[0] for err in ["No sources available", "Error", "Web search unavailable", "Configuration"])

# #         doc_sources = [s for s in sources if is_doc_source(s)]
# #         web_sources = [s for s in sources if is_web_source(s)]
# #         only_error_sources = all(is_error_source(s) for s in sources)

# #         if not self.web_search_enabled:
# #             if doc_sources:
# #                 source_info = "Based on your uploaded document:"
# #             elif only_error_sources:
# #                 source_info = "No relevant information found in your uploaded document."
# #             else:
# #                 source_info = "Based on your uploaded document:"
# #         else:
# #             if doc_sources and web_sources:
# #                 source_info = "Based on your uploaded document and web search results:"
# #             elif doc_sources:
# #                 source_info = "Based on your uploaded document:"
# #             elif web_sources:
# #                 source_info = "Based on web search results:"
# #             elif only_error_sources:
# #                 source_info = "No relevant information found in your uploaded document or web search."
# #             else:
# #                 source_info = "Based on your uploaded document:"

# #         prompt_text = f"""Answer the question using ONLY the information provided in the knowledge base.
# # DO NOT add information from your general knowledge or external sources.

# # STRICT GUIDELINES:
# # - Use ONLY the provided knowledge - nothing more
# # - Include ALL relevant information from the sources that answers the question
# # - If the knowledge is limited, be honest about limitations but provide what is available
# # - Do NOT elaborate beyond the source material
# # - Present information completely - don't omit parts of statements
# # - Keep responses focused but comprehensive based on available sources
# # - Do NOT invent details or explanations not in the sources

# # CRITICAL: For any information from uploaded documents, include source reference: [Source: DOCUMENT_NAME, page X, paragraph Y]

# # Query: {query}

# # {source_info}
# # Available Knowledge (use ONLY this information):
# # {knowledge}

# # Sources:
# # {sources_text}

# # Provide a complete answer using ALL the relevant information provided:

# # Answer:"""

# #         result = self._call_llm_with_retry(prompt_text)
        
# #         # Validate response against sources to prevent hallucination
# #         response_content = result.content
        
# #         # Basic validation: check response length vs source material
# #         if hasattr(self, '_last_source_chunks') and self._last_source_chunks:
# #             source_text = " ".join([chunk.get('text', '') for chunk in self._last_source_chunks])
# #             source_words = len(source_text.split())
# #             response_words = len(response_content.split())
            
# #             if response_words > source_words * 0.5:
# #                 print(f"⚠️ Warning: Response ({response_words} words) may exceed source material ({source_words} words)")
# #                 print("   Response may contain hallucinated information")
        
# #         # Format source references with color for better readability
# #         formatted_response = self._format_sources_with_color(response_content)
        
# #         return formatted_response

# # Function to validate command line inputs
# def validate_args(args):
#     if args.max_tokens <= 0:
#         raise ValueError("max_tokens must be a positive integer.")
#     if args.temperature < 0 or args.temperature > 1:
#         raise ValueError("temperature must be between 0 and 1.")
#     return args


# # Function to parse command line arguments
# def parse_args():
#     parser = argparse.ArgumentParser(description="CRAG Process for Document Retrieval and Query Answering.")
#     parser.add_argument("--file_path", type=str, default="data/Understanding_Climate_Change (1).pdf",
#                         help="Path to the document file to encode (supports PDF, TXT, CSV, JSON, DOCX, XLSX).")
#     parser.add_argument("--model", type=str, default="gemini-1.5-flash",
#                         help="Language model to use (default: gemini-1.5-flash).")
#     parser.add_argument("--max_tokens", type=int, default=1000,
#                         help="Maximum tokens to use in LLM responses (default: 1000).")
#     parser.add_argument("--temperature", type=float, default=0,
#                         help="Temperature to use for LLM responses (default: 0).")
#     parser.add_argument("--query", type=str, default="What are the main causes of climate change?",
#                         help="Query to test the CRAG process.")
#     parser.add_argument("--lower_threshold", type=float, default=0.3,
#                         help="Lower threshold for score evaluation (default: 0.3).")
#     parser.add_argument("--upper_threshold", type=float, default=0.7,
#                         help="Upper threshold for score evaluation (default: 0.7).")
#     parser.add_argument("--web_search_enabled", type=str, default=None,
#                         help="Set to 'true' to enable web search, 'false' to disable, or leave unset to use env var.")

#     args = parser.parse_args()
#     # Convert web_search_enabled to bool or None
#     if args.web_search_enabled is not None:
#         val = args.web_search_enabled.strip().lower()
#         if val == 'true':
#             args.web_search_enabled = True
#         elif val == 'false':
#             args.web_search_enabled = False
#         else:
#             args.web_search_enabled = None
#     return validate_args(args)


# # Main function to handle argument parsing and call the CRAG class
# def main(args):
#     # Initialize the CRAG process
#     crag = CRAG(
#         file_path=args.file_path,
#         model=args.model,
#         max_tokens=args.max_tokens,
#         temperature=args.temperature,
#         lower_threshold=args.lower_threshold,
#         upper_threshold=args.upper_threshold,
#         web_search_enabled=args.web_search_enabled
#     )

#     # Process the query
#     response = crag.run(args.query)
#     print(f"Query: {args.query}")
#     print(f"Answer: {response}")






# # If running as a script, launch Streamlit UI if desired
# if __name__ == '__main__':
#     import sys

#     main(parse_args())