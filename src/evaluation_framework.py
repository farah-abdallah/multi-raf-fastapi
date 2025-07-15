"""
Evaluation Framework for Multi-RAG Chatbot

This module provides comprehensive evaluation capabilities for RAG techniques including:
- User feedback collection (Phase 1)
- Automated evaluation metrics (Phase 2)
- Response scoring and storage (Phase 3)
- Analytics and comparison tools (Phase 4)
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic similarity evaluation will be disabled.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google GenerativeAI not available. LLM-based evaluation will be disabled.")

# Import API key manager for rotation
try:
    from api_key_manager import get_api_manager
    API_MANAGER_AVAILABLE = True
    print("🔑 Evaluation Framework API key rotation manager loaded")
except ImportError:
    API_MANAGER_AVAILABLE = False
    print("⚠️ Evaluation Framework API key manager not found - using single key mode")

@dataclass
class EvaluationMetrics:
    """Data class for storing evaluation metrics"""
    relevance_score: float = 0.0
    faithfulness_score: float = 0.0
    completeness_score: float = 0.0
    semantic_similarity: float = 0.0
    response_length: int = 0
    processing_time: float = 0.0
    timestamp: str = ""

@dataclass
class UserFeedback:
    """Data class for storing user feedback"""
    helpfulness: int = 0  # 1-5 scale
    accuracy: int = 0     # 1-5 scale
    clarity: int = 0      # 1-5 scale
    overall_rating: int = 0  # 1-5 scale
    comments: str = ""
    timestamp: str = ""

@dataclass
class EvaluationResult:
    """Complete evaluation result combining automated metrics and user feedback"""
    query_id: str
    query: str
    response: str
    technique: str
    document_sources: List[str]
    automated_metrics: EvaluationMetrics
    user_feedback: Optional[UserFeedback] = None
    session_id: str = ""

class EvaluationDatabase:
    """Database manager for storing evaluation results"""
    
    def __init__(self, db_path: str = "rag_evaluation.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the evaluation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                query_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                technique TEXT NOT NULL,
                document_sources TEXT,
                session_id TEXT,
                
                -- Automated metrics
                relevance_score REAL,
                faithfulness_score REAL,
                completeness_score REAL,
                semantic_similarity REAL,
                response_length INTEGER,
                processing_time REAL,
                metrics_timestamp TEXT,
                
                -- User feedback
                helpfulness INTEGER,
                accuracy INTEGER,
                clarity INTEGER,
                overall_rating INTEGER,
                feedback_comments TEXT,
                feedback_timestamp TEXT,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_evaluation(self, result: EvaluationResult):
        """Store evaluation result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO evaluations (
                query_id, query, response, technique, document_sources, session_id,
                relevance_score, faithfulness_score, completeness_score, 
                semantic_similarity, response_length, processing_time, metrics_timestamp,
                helpfulness, accuracy, clarity, overall_rating, 
                feedback_comments, feedback_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.query_id, result.query, result.response, result.technique,
            json.dumps(result.document_sources), result.session_id,
            result.automated_metrics.relevance_score,
            result.automated_metrics.faithfulness_score,
            result.automated_metrics.completeness_score,
            result.automated_metrics.semantic_similarity,
            result.automated_metrics.response_length,
            result.automated_metrics.processing_time,
            result.automated_metrics.timestamp,
            result.user_feedback.helpfulness if result.user_feedback else None,
            result.user_feedback.accuracy if result.user_feedback else None,
            result.user_feedback.clarity if result.user_feedback else None,
            result.user_feedback.overall_rating if result.user_feedback else None,
            result.user_feedback.comments if result.user_feedback else None,
            result.user_feedback.timestamp if result.user_feedback else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_technique_performance(self, technique: str = None) -> List[Dict]:
        """Get performance statistics for techniques"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if technique:
            cursor.execute("""
                SELECT * FROM evaluations WHERE technique = ?
                ORDER BY created_at DESC
            """, (technique,))
        else:
            cursor.execute("""
                SELECT * FROM evaluations
                ORDER BY created_at DESC
            """)
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_comparison_data(self) -> Dict[str, Any]:
        """Get aggregated data for technique comparison"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get average metrics by technique
        cursor.execute("""
            SELECT 
                technique,
                COUNT(*) as total_queries,
                AVG(relevance_score) as avg_relevance,
                AVG(faithfulness_score) as avg_faithfulness,
                AVG(completeness_score) as avg_completeness,
                AVG(semantic_similarity) as avg_semantic_similarity,
                AVG(processing_time) as avg_processing_time,
                AVG(response_length) as avg_response_length,
                AVG(CASE WHEN overall_rating IS NOT NULL THEN overall_rating END) as avg_user_rating,
                COUNT(CASE WHEN overall_rating IS NOT NULL THEN 1 END) as feedback_count
            FROM evaluations
            GROUP BY technique
        """)
        
        comparison_data = {}
        columns = [description[0] for description in cursor.description]
        
        for row in cursor.fetchall():
            technique_data = dict(zip(columns, row))
            comparison_data[technique_data['technique']] = technique_data
        
        conn.close()
        return comparison_data

class AutomatedEvaluator:
    """Automated evaluation using LLMs and embeddings"""
    
    def __init__(self):
        self.embedding_model = None
        self.gemini_model = None
        self.use_llm_evaluation = False
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Embedding model loaded successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not load embedding model: {e}")
          # Initialize Gemini with API key management
        if GEMINI_AVAILABLE:
            try:
                if API_MANAGER_AVAILABLE:                    # Use API manager if available
                    self.api_manager = get_api_manager()
                    print(f"🎯 Evaluation Framework API Manager Status: {self.api_manager.get_status()}")
                    current_key = self.api_manager.get_current_key()
                    genai.configure(api_key=current_key)
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    self.use_llm_evaluation = True
                    self.use_api_rotation = True
                    print("✅ Gemini model initialized with API key rotation")
                else:                    # Fallback to single key
                    api_key = os.getenv('GOOGLE_API_KEY')
                    if api_key:
                        genai.configure(api_key=api_key)
                        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                        self.use_llm_evaluation = True
                        self.use_api_rotation = False
                        print("✅ Gemini model initialized with single API key")
                    else:
                        print("⚠️ GOOGLE_API_KEY not found. Using heuristic evaluation instead of LLM.")
                        self.use_llm_evaluation = False
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Gemini: {e}")
                self.use_llm_evaluation = False
        else:
            print("⚠️ Gemini not available. Using heuristic evaluation.")
            self.use_llm_evaluation = False
    
    def _generate_with_rotation(self, prompt: str, max_retries: int = 3):
        """Generate content with automatic API key rotation on quota errors"""
        if not self.use_llm_evaluation:
            raise Exception("LLM evaluation not available")
        
        if not self.use_api_rotation:
            # Single key mode
            return self.gemini_model.generate_content(prompt)
        
        # Key rotation mode
        for attempt in range(max_retries):
            try:
                # Get current API key and configure genai
                current_key = self.api_manager.get_current_key()
                genai.configure(api_key=current_key)
                  # Recreate model with new API key
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                response = self.gemini_model.generate_content(prompt)
                
                return response
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for quota/rate limit errors
                if any(err in error_msg for err in ['quota', 'rate limit', '429', 'resource_exhausted']):
                    print(f"⚠️ Evaluation Framework Quota/rate limit hit on attempt {attempt + 1}: {e}")
                    
                    if attempt < max_retries - 1:  # Not the last attempt
                        if self.api_manager.rotate_key():
                            print(f"🔄 Evaluation Framework Rotated to new API key, retrying...")
                            continue
                        else:
                            print("❌ Evaluation Framework No more API keys available for rotation")
                            raise Exception("All API keys exhausted due to quota limits")
                    else:
                        raise Exception(f"Max retries ({max_retries}) exceeded due to quota limits")
                else:
                    # Non-quota error, don't retry
                    raise e
        
        raise Exception(f"Failed to generate content after {max_retries} attempts")
    
    def evaluate_relevance_heuristic(self, query: str, response: str) -> float:
        """Heuristic relevance evaluation based on keyword overlap and length"""
        try:
            # Simple keyword-based relevance
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(response_words))
            union = len(query_words.union(response_words))
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = overlap / union
            
            # Factor in response length (too short = likely incomplete, too long = likely unfocused)
            response_length = len(response.split())
            optimal_length = len(query.split()) * 10  # Heuristic: 10x query length
            length_factor = min(1.0, response_length / optimal_length) if optimal_length > 0 else 0.5
              # Combine factors
            relevance_score = (jaccard_similarity * 0.7) + (length_factor * 0.3)
            return min(1.0, relevance_score)
            
        except Exception as e:
            print(f"Error in heuristic relevance evaluation: {e}")
            return 0.5

    def evaluate_faithfulness_heuristic(self, context: str, response: str) -> float:
        """Heuristic faithfulness evaluation based on context overlap"""
        try:
            if not context or len(context.strip()) == 0:
                # Without context, we can't really evaluate faithfulness
                return 0.3
            
            # Check for placeholder contexts that don't contain actual content
            placeholder_indicators = [
                'context from uploaded documents',
                'sample document context',
                'crag retrieved context',
                'no specific document context',
                'context not available'
            ]
            
            context_lower = context.lower()
            if any(placeholder in context_lower for placeholder in placeholder_indicators):
                # This is a placeholder context, return a low score
                return 0.4
            
            # Clean and tokenize
            response_lower = response.lower()
            
            # Remove common stop words for better matching
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
            
            context_words = set(word for word in context_lower.split() if word not in stop_words and len(word) > 2)
            response_words = set(word for word in response_lower.split() if word not in stop_words and len(word) > 2)
            
            if len(response_words) == 0:
                return 0.1  # Empty response
            
            if len(context_words) == 0:
                # Context has no meaningful words
                return 0.2
            
            # Words from response that appear in context
            supported_words = response_words.intersection(context_words)
            support_ratio = len(supported_words) / len(response_words)
            
            # Check for potential hallucinations - words in response not in context
            unsupported_words = response_words - context_words
            hallucination_ratio = len(unsupported_words) / len(response_words)
            
            # Apply penalties for hallucinations
            if hallucination_ratio > 0.7:
                support_ratio *= 0.5  # Heavy penalty for mostly unsupported content
            elif hallucination_ratio > 0.5:
                support_ratio *= 0.7  # Moderate penalty
            
            # More realistic scoring ranges
            if support_ratio > 0.8:
                return min(0.85, 0.65 + support_ratio * 0.2)  # Cap very high scores
            elif support_ratio > 0.6:
                return 0.55 + support_ratio * 0.2  # Good faithfulness: 0.67-0.71
            elif support_ratio > 0.4:
                return 0.35 + support_ratio * 0.4  # Moderate: 0.51-0.59
            elif support_ratio > 0.2:
                return 0.25 + support_ratio * 0.5  # Low: 0.35-0.45
            else:
                return max(0.15, support_ratio * 1.5)  # Very low faithfulness
                
        except Exception as e:
            print(f"Error in heuristic faithfulness evaluation: {e}")
            return 0.3
    
    def evaluate_completeness_heuristic(self, query: str, response: str) -> float:
        """Heuristic completeness evaluation based on response characteristics"""
        try:
            query_length = len(query.split())
            response_length = len(response.split())
            
            # Very short responses are likely incomplete
            if response_length < 5:
                return 0.1
            elif response_length < 15:
                return 0.3
            
            # Check for question words in query to determine expected response depth
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'describe', 'compare', 'analyze', 'discuss']
            query_lower = query.lower()
            question_complexity = sum(1 for word in question_words if word in query_lower)
            
            # More complex questions need more detailed answers
            if question_complexity == 0:
                # Simple query, shorter response might be complete
                expected_min_length = 10
            else:
                expected_min_length = max(20, question_complexity * 12)
            
            # Check for completeness indicators
            completion_indicators = ['therefore', 'conclusion', 'summary', 'overall', 'finally', 'in summary']
            has_conclusion = any(indicator in response.lower() for indicator in completion_indicators)
            
            # Check for partial indicators that suggest incompleteness
            partial_indicators = ['however', 'but', 'although', 'more research', 'further', 'additional']
            has_partial = any(indicator in response.lower() for indicator in partial_indicators)
            
            # Base score from length
            if response_length >= expected_min_length * 1.5:
                base_score = 0.8  # Good length
            elif response_length >= expected_min_length:
                base_score = 0.6  # Adequate length
            else:
                base_score = 0.3 + (response_length / expected_min_length) * 0.3  # Proportional to length
            
            # Adjust based on content indicators
            if has_conclusion:
                base_score = min(1.0, base_score + 0.15)
            if has_partial:
                base_score = max(0.1, base_score - 0.1)
              # Check if response seems to address the query type
            if 'explain' in query_lower or 'how' in query_lower:
                # Explanatory questions need step-by-step or detailed responses
                step_indicators = ['first', 'second', 'then', 'next', 'finally', 'step', 'process']
                has_steps = any(indicator in response.lower() for indicator in step_indicators)
                if has_steps:
                    base_score = min(1.0, base_score + 0.1)
                else:
                    base_score = max(0.2, base_score - 0.2)
            
            return min(0.95, max(0.1, base_score))  # Cap at 0.95 to avoid perfect scores
        except Exception as e:
            print(f"Error in heuristic completeness evaluation: {e}")
            return 0.4
    
    def evaluate_relevance(self, query: str, response: str) -> float:
        """Evaluate how relevant the response is to the query using LLM or heuristics"""
        if not self.use_llm_evaluation or not self.gemini_model:
            return self.evaluate_relevance_heuristic(query, response)
        
        try:
            prompt = f"""
            Evaluate how relevant this response is to the given query on a scale of 0.0 to 1.0.
            
            Query: {query}
            Response: {response}
            
            Consider:
            - Does the response directly address the query?
            - Is the information provided relevant to what was asked?
            - How well does the response match the intent of the query?
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            result = self._generate_with_rotation(prompt)
            score = float(result.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Error in LLM relevance evaluation, falling back to heuristic: {e}")
            return self.evaluate_relevance_heuristic(query, response)
    
    def evaluate_faithfulness(self, context: str, response: str) -> float:
        """Evaluate if the response is faithful to the source context using LLM or heuristics"""
        if not self.use_llm_evaluation or not self.gemini_model:
            return self.evaluate_faithfulness_heuristic(context, response)
        
        try:
            # Truncate context for better evaluation
            context_truncated = context[:800] if len(context) > 800 else context
            
            prompt = f"""Evaluate how faithful this response is to the given context on a scale of 0.0 to 1.0.

Context: {context_truncated}

Response: {response}

Detailed scoring criteria:
- 0.9-1.0: Response is comprehensive, highly detailed, and fully supported by context
- 0.7-0.8: Response is accurate and well-supported, with good detail from context
- 0.5-0.6: Response is partially accurate but lacks detail or has some unsupported claims
- 0.3-0.4: Response has minimal connection to context, mostly general statements
- 0.1-0.2: Response barely relates to context or contains unsupported information
- 0.0: Response contradicts context or has no factual basis

Consider:
- Factual accuracy and consistency with context
- Level of detail and specificity from the context
- Completeness of information coverage
- Whether all claims are supported by the provided context

Respond with ONLY a decimal number between 0.0 and 1.0 (e.g., 0.85)."""
            
            result = self._generate_with_rotation(prompt)
            
            # Clean the result and extract number
            result_text = result.text.strip()
            print(f"Debug - LLM faithfulness result: '{result_text}'")
            
            # Extract the first number found in the response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', result_text)
            if numbers:
                score = float(numbers[0])
                score = max(0.0, min(1.0, score))
                print(f"Debug - Parsed faithfulness score: {score}")
                return score
            else:
                print(f"Debug - No valid number found, falling back to heuristic")
                return self.evaluate_faithfulness_heuristic(context, response)
            
        except Exception as e:
            print(f"Error in LLM faithfulness evaluation, falling back to heuristic: {e}")
            return self.evaluate_faithfulness_heuristic(context, response)
    
    def evaluate_completeness(self, query: str, response: str) -> float:
        """Evaluate how complete the response is using LLM or heuristics"""
        if not self.use_llm_evaluation or not self.gemini_model:
            return self.evaluate_completeness_heuristic(query, response)
        
        try:
            prompt = f"""
            Evaluate how complete this response is in answering the query on a scale of 0.0 to 1.0.
            
            Query: {query}
            Response: {response}
            
            Consider:
            - Does the response fully address all aspects of the query?
            - Are there important missing pieces of information?            - Is the level of detail appropriate for the question?
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            result = self._generate_with_rotation(prompt)
            score = float(result.text.strip())
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Error in LLM completeness evaluation, falling back to heuristic: {e}")
            return self.evaluate_completeness_heuristic(query, response)
    
    def evaluate_semantic_similarity(self, query: str, response: str) -> float:
        """Evaluate semantic similarity between query and response using embeddings"""
        if not self.embedding_model:
            return 0.5
        
        try:
            query_embedding = self.embedding_model.encode([query])
            response_embedding = self.embedding_model.encode([response])
            
            similarity = cosine_similarity(query_embedding, response_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Error in semantic similarity evaluation: {e}")
            return 0.5
    
    def evaluate_response(self, query: str, response: str, context: str = "", processing_time: float = 0.0) -> EvaluationMetrics:
        """Perform comprehensive automated evaluation"""
        metrics = EvaluationMetrics()
          # Calculate basic metrics
        metrics.response_length = len(response) if response else 0
        metrics.processing_time = processing_time
        metrics.timestamp = datetime.now().isoformat()
        
        # Calculate advanced metrics if models are available
        if self.gemini_model or self.embedding_model:
            metrics.relevance_score = self.evaluate_relevance(query, response)
            # Always evaluate faithfulness, even with empty context (our heuristic handles this)
            metrics.faithfulness_score = self.evaluate_faithfulness(context, response)
            metrics.completeness_score = self.evaluate_completeness(query, response)
            metrics.semantic_similarity = self.evaluate_semantic_similarity(query, response)
        
        return metrics

class EvaluationManager:
    """Main evaluation manager coordinating all evaluation activities"""
    
    def __init__(self, db_path: str = "rag_evaluation.db"):
        self.database = EvaluationDatabase(db_path)
        self.evaluator = AutomatedEvaluator()
    
    def evaluate_rag_response(self, query: str, response: str, technique: str, 
                            document_sources: List[str], context: str = "",
                            processing_time: float = 0.0, session_id: str = "") -> str:
        """Evaluate a RAG response and store results"""
        # Generate unique query ID
        query_id = hashlib.md5(f"{query}{response}{technique}{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Perform automated evaluation
        automated_metrics = self.evaluator.evaluate_response(query, response, context, processing_time)
        
        # Create evaluation result
        result = EvaluationResult(
            query_id=query_id,
            query=query,
            response=response,
            technique=technique,
            document_sources=document_sources,
            automated_metrics=automated_metrics,
            session_id=session_id
        )
        
        # Store in database
        self.database.store_evaluation(result)
        
        return query_id
    
    def add_user_feedback(self, query_id: str, feedback: UserFeedback):
        """Add user feedback to an existing evaluation"""
        # Get existing evaluation
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM evaluations WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        
        if row:
            # Update with user feedback
            cursor.execute("""
                UPDATE evaluations SET
                    helpfulness = ?, accuracy = ?, clarity = ?, 
                    overall_rating = ?, feedback_comments = ?, feedback_timestamp = ?
                WHERE query_id = ?
            """, (
                feedback.helpfulness, feedback.accuracy, feedback.clarity,
                feedback.overall_rating, feedback.comments, feedback.timestamp,
                query_id
            ))
            
            conn.commit()
        
        conn.close()
    
    def delete_evaluation(self, query_id: str):
        """Delete an evaluation record by query_id"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evaluations WHERE query_id = ?", (query_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error deleting evaluation: {e}")
            return False
    
    def get_technique_comparison(self) -> Dict[str, Any]:
        """Get comprehensive comparison data for all techniques"""
        return self.database.get_comparison_data()
    
    def get_performance_history(self, technique: str = None, limit: int = 100) -> List[Dict]:
        """Get performance history for analysis"""
        return self.database.get_technique_performance(technique)[:limit]
    
    def export_evaluation_data(self, filename: str = None) -> str:
        """Export all evaluation data to JSON"""
        if not filename:
            filename = f"rag_evaluation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        comparison_data = self.get_technique_comparison()
        performance_history = self.get_performance_history()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "technique_comparison": comparison_data,
            "performance_history": performance_history,
            "summary": {
                "total_evaluations": len(performance_history),
                "techniques_evaluated": list(comparison_data.keys()),
                "evaluation_period": {
                    "start": performance_history[-1]["created_at"] if performance_history else None,
                    "end": performance_history[0]["created_at"] if performance_history else None
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    def add_query_response(self, query_id: str, query: str, response: str, context: str = "", 
                          technique: str = "", metadata: Dict = None) -> str:
        """Add a query/response pair to the evaluation database without full evaluation
        
        This is a lighter version of evaluate_rag_response for simple logging.
        """
        # Create basic metrics with current timestamp
        basic_metrics = EvaluationMetrics(
            response_length=len(response) if response else 0,
            processing_time=0.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Create evaluation result with basic info
        result = EvaluationResult(
            query_id=query_id,
            query=query,
            response=response,
            technique=technique,
            document_sources=metadata.get('source_chunks', []) if metadata else [],
            automated_metrics=basic_metrics,
            session_id=""
        )
        
        # Store in database
        self.database.store_evaluation(result)
        
        return query_id
