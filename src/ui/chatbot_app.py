"""
Multi-RAG Chatbot FastAPI Application with Document Upload, Message History, and Comprehensive Evaluation

This application provides a web-based interface for testing different RAG techniques:
- Adaptive RAG
- CRAG (Corrective RAG)
- Document Augmentation RAG
- Basic RAG
- Explainable Retrieval RAG

Features:
- Upload documents (PDF, CSV, TXT, JSON, DOCX, XLSX)
- Choose RAG technique from dropdown
- Message history with technique tracking
- Comprehensive evaluation framework with user feedback and automated metrics
- Analytics dashboard for comparing technique performance
- Elegant, responsive UI
"""
import sys
import os
import tempfile
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
import sqlite3
import asyncio
from pathlib import Path

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import mimetypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import our RAG systems
from src.adaptive_rag import AdaptiveRAG
from src.crag import CRAG
from src.document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings, load_document_content
from src.utils.helpers import encode_document, replace_t_with_space
from src.explainable_retrieval import ExplainableRAGMethod

# Import evaluation framework
from src.evaluation_framework import EvaluationManager, UserFeedback
from src.analytics_dashboard import display_analytics_dashboard

# Import document viewer components
from src.document_viewer import create_document_link, show_embedded_document_viewer, check_document_viewer_page

# Templates directory
templates = Jinja2Templates(directory="src/ui/templates")

# Define lifespan handler (moved up from below)
from contextlib import asynccontextmanager

evaluation_manager = None

@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for FastAPI"""
    # Startup code
    global evaluation_manager
    init_chat_database()
    evaluation_manager = EvaluationManager()
    print("FastAPI Multi-RAG Chatbot started successfully!")
    yield
    # Shutdown code (if any)

# FastAPI app instance with lifespan
app = FastAPI(
    title="Multi-RAG Chatbot", 
    description="A comprehensive RAG evaluation platform",
    lifespan=lifespan
)

# Static files directory for CSS, JS, etc.
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")

# Global variables to replace st.session_state
sessions = {}
evaluation_manager = None
rag_systems = {}

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    technique: str
    session_id: str
    crag_web_search_enabled: bool = True

class FeedbackRequest(BaseModel):
    query_id: str
    helpfulness: int
    accuracy: int
    clarity: int
    overall_rating: int
    comments: str = ""

class SessionData(BaseModel):
    session_id: str
    messages: List[Dict]
    uploaded_documents: List[str] = []
    selected_technique: str = "Adaptive RAG"
    crag_web_search_enabled: bool = True
    pending_feedback: Dict[str, bool] = {}
    last_source_chunks: Dict[str, List] = {}

# === DATABASE FUNCTIONS (same as Streamlit version) ===

def init_chat_database():
    """Initialize the chat history database"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                technique TEXT,
                query_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print("Chat database initialized successfully")
    except Exception as e:
        print(f"Error initializing chat database: {e}")

def save_chat_message(session_id: str, message_id: str, message_type: str, content: str, technique: str = None, query_id: str = None):
    """Save a chat message to the database"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_sessions (session_id, message_id, message_type, content, technique, query_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, message_id, message_type, content, technique, query_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat message: {e}")

def load_chat_history(session_id: str):
    """Load chat history for a specific session"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT message_id, message_type as role, content, technique, query_id, timestamp
            FROM chat_sessions 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            message_id, role, content, technique, query_id, timestamp = row
            message={
                "id": message_id,
                "role": role,
                "content": content,
                #"technique": technique,
                #"query_id": query_id,
                "timestamp": timestamp
            }
            if technique:
                message["technique"] = technique
            if query_id:
                message["query_id"] = query_id
            messages.append(message)
        
        conn.close()
        return messages
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []
    

import uuid
from datetime import datetime
import sqlite3
import os

def get_or_create_session_id(session: dict) -> str:
    """Get existing session ID or create new one (for FastAPI, use session dict)"""
    if 'persistent_session_id' not in session:
        session['persistent_session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    return session['persistent_session_id']

def auto_save_chat(session: dict, messages: list, save_chat_message_func):
    """Auto-save current chat messages (for FastAPI, pass session and messages)"""
    if messages:
        session_id = get_or_create_session_id(session)
        if 'last_saved_count' not in session:
            session['last_saved_count'] = 0
        new_messages = messages[session['last_saved_count']:]
        for msg in new_messages:
            save_chat_message_func(
                session_id=session_id,
                message_id=msg.get('id', str(uuid.uuid4())),
                message_type=msg['role'],
                content=msg['content'],
                technique=msg.get('technique'),
                query_id=msg.get('query_id')
            )
        session['last_saved_count'] = len(messages)

def clear_current_session(session: dict):
    """Clear current session chat history (for FastAPI, use session dict)"""
    session_id = get_or_create_session_id(session)
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
        # Clear session state
        session['messages'] = []
        session['last_saved_count'] = 0
    except Exception as e:
        print(f"Error clearing session: {e}")

def delete_chat_message(session_id: str, message_id: str):
    """Delete a specific chat message from database"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ? AND message_id = ?', (session_id, message_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error deleting chat message: {e}")

def delete_conversation_pair(session: dict, user_message_id: str, assistant_message_id: str = None, query_id: str = None):
    """Delete a conversation pair (user question + assistant response) and associated ratings"""
    session_id = get_or_create_session_id(session)
    try:
        # Delete from chat history database
        delete_chat_message(session_id, user_message_id)
        if assistant_message_id:
                delete_chat_message(session_id, assistant_message_id)
        # Delete from evaluation database if query_id exists
        if query_id :
            evaluation_manager=get_evaluation_manager()
            evaluation_manager.delete_evaluation(query_id)
        # Remove from session state
        message_ids_to_delete = [user_message_id]
        if assistant_message_id:
            message_ids_to_delete.append(assistant_message_id)

        session['messages'] = [
            msg for msg in session.get('messages', [])
            if msg.get('id') not in message_ids_to_delete
        ]

        #update last saved count
        session['last_saved_count'] = len(session['messages'])
        return True
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False

def get_evaluation_manager():
    """Get the global evaluation manager instance"""
    global evaluation_manager
    if evaluation_manager is None:
        evaluation_manager = EvaluationManager()
    return evaluation_manager

def initialize_session_state(session: dict):
    """Initialize session state variables with persistent chat history (for FastAPI, use session dict)"""
    #initialize chat database
    init_chat_database()
    # Get or create persistent session ID
    session_id = get_or_create_session_id(session)
    # Initialize messages with persistent storage
    if 'messages' not in session:
        saved_messages = load_chat_history(session_id)
        session['messages'] = saved_messages if saved_messages else []
        session['last_saved_count'] = len(session['messages'])
    if 'uploaded_documents' not in session:
        session['uploaded_documents'] = []
    if 'rag_systems' not in session:
        session['rag_systems'] = {}
    if 'document_content' not in session:
        session['document_content'] = None
    if 'last_document_hash' not in session:
        session['last_document_hash'] = None
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if 'pending_feedback' not in session:
        session['pending_feedback'] = {}
    if 'current_page' not in session:
        session['current_page'] = "Chat"
    if 'last_source_chunks' not in session:
        session['last_source_chunks'] = {}
    #update last activity timestamp
    session['last_activity'] = datetime.now()

def get_all_chat_sessions():
    """Get all chat sessions with their first message as title"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, MIN(timestamp) as first_message_time,
                   (SELECT content FROM chat_sessions cs2 
                    WHERE cs2.session_id = cs1.session_id 
                    AND cs2.message_type = 'user' 
                    ORDER BY timestamp ASC LIMIT 1) as first_message
            FROM chat_sessions cs1
            GROUP BY session_id
            ORDER BY first_message_time DESC
        ''')
        
        sessions = []
        for row in cursor.fetchall():
            session_id, timestamp, first_message = row
            # Create a readable title from the first message
            if first_message:
                title = first_message[:50] + "..." if len(first_message) > 50 else first_message
            else:
                title = f"Chat {session_id[-8:]}"
            
            sessions.append({
                'session_id': session_id,
                'title': title,
                'timestamp': timestamp,
                'first_message': first_message
            })
        
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error getting chat sessions: {e}")
        return []

def delete_chat_session(session_id: str):
    """Delete an entire chat session (FastAPI version)"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
        
        # If we're deleting the current session, create a new one
        if session_id in sessions and sessions[session_id].session_id == session_id:
            del sessions[session_id]
            new_session_id = create_new_chat_session()
            return True, new_session_id
        else:
            if session_id in sessions:
                del sessions[session_id]
            return True, None
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return False, None

def rename_chat_session(session_id: str, new_title: str):
    """Rename a chat session by updating its first message"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        # Update the first user message content to serve as the title
        cursor.execute('''
            UPDATE chat_sessions 
            SET content = ?
            WHERE session_id = ?
            AND message_type = 'user'
            AND timestamp = (
                SELECT MIN(timestamp) 
                FROM chat_sessions 
                WHERE session_id = ? AND message_type = 'user'
            )
        ''', (new_title, session_id, session_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error renaming chat session: {e}")
        return False
        

# === RAG SYSTEM FUNCTIONS (adapted from Streamlit version) ===

def save_uploaded_file(uploaded_file: UploadFile):
    """Save uploaded file to temporary directory and return path (FastAPI version)"""
    import tempfile
    import os

    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.filename)  # Use .filename for FastAPI UploadFile

        with open(file_path, "wb") as f:
            content = uploaded_file.file.read()  # .file is a SpooledTemporaryFile
            f.write(content)

        return file_path
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None

def load_rag_system(
    technique: str,
    document_paths: List[str] = None,
    crag_web_search_enabled: bool = None
):
    """Load the specified RAG system with error handling and collect UI messages."""
    messages = []
    rag_system = None
    try:
        messages.append({"type": "info", "text": f"Loading {technique}..."})
        if technique == "Adaptive RAG":
            if document_paths:
                rag_system = AdaptiveRAG(file_paths=document_paths)
            else:
                rag_system = AdaptiveRAG(texts=["Sample text for testing. This is a basic RAG system."])
            messages.append({"type": "success", "text": "Adaptive RAG loaded successfully."})

        elif technique == "CRAG":
            if document_paths and len(document_paths) > 0:
                rag_system = CRAG(document_paths, web_search_enabled=crag_web_search_enabled)
                messages.append({"type": "success", "text": "CRAG loaded with your documents."})
            else:
                sample_file = "data/Understanding_Climate_Change (1).pdf"
                if os.path.exists(sample_file):
                    rag_system = CRAG([sample_file], web_search_enabled=crag_web_search_enabled)
                    messages.append({"type": "warning", "text": "No documents uploaded. Using sample file."})
                else:
                    messages.append({"type": "error", "text": "CRAG requires a document. Please upload a file first."})

        elif technique == "Document Augmentation":
            if document_paths and len(document_paths) > 0:
                messages.append({"type": "info", "text": f"Processing {len(document_paths)} document(s) for Document Augmentation..."})
                combined_content = ""
                processed_docs = []
                for doc_path in document_paths:
                    try:
                        content = load_document_content(doc_path)
                        doc_name = os.path.basename(doc_path)
                        combined_content += f"\n\n=== Document: {doc_name} ===\n{content}"
                        processed_docs.append(doc_name)
                        messages.append({"type": "success", "text": f"✅ Processed: {doc_name}"})
                    except Exception as e:
                        messages.append({"type": "warning", "text": f"⚠️ Skipped {os.path.basename(doc_path)}: {str(e)}"})
                if combined_content:
                    messages.append({"type": "info", "text": f"Combined content from {len(processed_docs)} documents"})
                    embedding_model = SentenceTransformerEmbeddings()
                    processor = DocumentProcessor(combined_content, embedding_model, document_paths[0])
                    rag_system = processor.run()
                else:
                    messages.append({"type": "error", "text": "No documents could be processed successfully."})
            else:
                messages.append({"type": "error", "text": "Document Augmentation requires a document. Please upload a file first."})

        elif technique == "Basic RAG":
            if document_paths and len(document_paths) > 0:
                if len(document_paths) > 1:
                    messages.append({"type": "info", "text": f"Processing {len(document_paths)} documents with Basic RAG..."})
                    rag_system = create_multi_document_basic_rag(document_paths)
                else:
                    rag_system = encode_document(document_paths[0])
                messages.append({"type": "success", "text": "Basic RAG loaded successfully."})
            else:
                messages.append({"type": "error", "text": "Basic RAG requires a document. Please upload a file first."})

        elif technique == "Explainable Retrieval":
            if document_paths and len(document_paths) > 0:
                messages.append({"type": "info", "text": f"Processing {len(document_paths)} document(s) for Explainable Retrieval..."})
                all_texts = []
                processed_docs = []
                for doc_path in document_paths:
                    try:
                        content = load_document_content(doc_path)
                        doc_name = os.path.basename(doc_path)
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(content)
                        for chunk in chunks:
                            all_texts.append(f"[Source: {doc_name}] {chunk}")
                        processed_docs.append(doc_name)
                        messages.append({"type": "success", "text": f"✅ Processed: {doc_name}"})
                    except Exception as e:
                        messages.append({"type": "warning", "text": f"⚠️ Skipped {os.path.basename(doc_path)}: {str(e)}"})
                if all_texts:
                    messages.append({"type": "info", "text": f"Created explainable retrieval system with content from {len(processed_docs)} documents"})
                    rag_system = ExplainableRAGMethod(all_texts)
                else:
                    messages.append({"type": "error", "text": "No documents could be processed successfully."})
            else:
                messages.append({"type": "error", "text": "Explainable Retrieval requires a document. Please upload a file first."})

    except Exception as e:
        messages.append({"type": "error", "text": f"Error loading {technique}: {str(e)}"})

    return rag_system, messages

def get_rag_response(technique: str, query: str, rag_system, session_messages: list):
    """
    Get response from the specified RAG system and return response, context, source_chunks, and UI messages.
    """
    messages = []
    try:
        context = ""  # Will store retrieved context for evaluation
        source_chunks = None  # For CRAG responses

        if technique == "Adaptive RAG":
            conversation_context = ""
            if len(session_messages) > 1:
                conversation_context = build_conversation_context(session_messages)
                messages.append({"type": "info", "text": "Using conversation history for context-aware response..."})

            enhanced_query = query
            if conversation_context:
                enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
            try:
                context = rag_system.get_context_for_query(enhanced_query, silent=True)
                response = rag_system.answer(enhanced_query, silent=True)
                
                # Try to get source chunks for document viewer
                try:
                    docs = rag_system.get_relevant_documents(enhanced_query)
                    if docs:
                        source_chunks = []
                        for i, doc in enumerate(docs[:3]):
                            chunk = {
                                'text': doc.page_content,
                                'source': doc.metadata.get('source', 'Unknown'),
                                'page': doc.metadata.get('page'),
                                'paragraph': doc.metadata.get('paragraph', i),
                                'score': 1.0  # Default score for Adaptive RAG
                            }
                            source_chunks.append(chunk)
                        return response, context, source_chunks, messages
                except:
                    pass
                
                return response, context, None, messages
            except Exception as e:
                response = rag_system.answer(enhanced_query)
                try:
                    docs = rag_system.get_relevant_documents(enhanced_query)
                    if docs:
                        context = "\n".join([doc.page_content[:500] for doc in docs[:3]])
                    else:
                        context = "No specific document context retrieved"
                except:
                    context = "Context from uploaded documents"
                return response, context, None, messages

        elif technique == "CRAG":
            try:
                messages.append({"type": "info", "text": "🔄 Running CRAG analysis..."})
                messages.append({"type": "info", "text": "**CRAG Process:**"})
                messages.append({"type": "info", "text": "1. Retrieving documents from your uploaded files..."})
                messages.append({"type": "info", "text": "2. Evaluating relevance to your query..."})

                conversation_context = ""
                if len(session_messages) > 1:
                    conversation_context = build_conversation_context(session_messages)
                    messages.append({"type": "info", "text": "3. Considering conversation history for context-aware response..."})

                enhanced_query = query
                if conversation_context:
                    enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
                if hasattr(rag_system, 'run_with_sources'):
                    result_data = rag_system.run_with_sources(enhanced_query)
                    response = result_data['answer']
                    source_chunks = result_data['source_chunks']
                    sources = result_data['sources']
                    messages.append({"type": "success", "text": "✅ CRAG analysis completed"})
                    context = "\n".join([chunk['text'] for chunk in source_chunks])
                    return response, context, source_chunks, messages
                else:
                    result = rag_system.run(enhanced_query)
                    response = result
                    messages.append({"type": "success", "text": "✅ CRAG analysis completed"})
                    try:
                        docs = rag_system.vectorstore.similarity_search(query, k=3)
                        if docs:
                            context = "\n".join([doc.page_content[:500] for doc in docs])
                        else:
                            context = "CRAG used web search or external sources"
                    except:
                        context = str(result)[:1000] if result else "CRAG context not available"

                messages.append({"type": "info", "text": """
**How CRAG worked for this query:**
- If relevance was HIGH (>0.7): Used your uploaded document
- If relevance was LOW (<0.3): Performed web search instead  
- If relevance was MEDIUM (0.3-0.7): Combined both sources

Check the response to see which source(s) were actually used!
"""})
                return response, context, None, messages
            except Exception as crag_error:
                error_msg = str(crag_error)
                messages.append({"type": "error", "text": f"❌ CRAG Error: {error_msg}"})
                if "API" in error_msg or "google" in error_msg.lower():
                    messages.append({"type": "warning", "text": "⚠️ This appears to be a Google API issue. Check your internet connection and API key."})
                elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    messages.append({"type": "warning", "text": "⚠️ API rate limit reached. Please wait a moment and try again."})
                return f"CRAG failed with error: {error_msg}", "", None, messages

        elif technique == "Document Augmentation":
            conversation_context = ""
            if len(session_messages) > 1:
                conversation_context = build_conversation_context(session_messages)
                messages.append({"type": "info", "text": "Using conversation history for context-aware response..."})

            enhanced_query = query
            if conversation_context:
                enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
            docs = rag_system.get_relevant_documents(enhanced_query)
            if docs:
                from src.document_augmentation import generate_answer
                context = docs[0].metadata.get('text', docs[0].page_content)
                response = generate_answer(context, enhanced_query)
                
                # Convert docs to source chunks format for document viewer
                source_chunks = []
                for i, doc in enumerate(docs[:3]):
                    chunk = {
                        'text': doc.page_content,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'page': doc.metadata.get('page'),
                        'paragraph': doc.metadata.get('paragraph', i),
                        'score': 1.0  # Default score for Document Augmentation
                    }
                    source_chunks.append(chunk)
                
                return response, context, source_chunks, messages
            else:
                return "No relevant documents found.", "", None, messages

        elif technique == "Basic RAG":
            conversation_context = ""
            if len(session_messages) > 1:
                conversation_context = build_conversation_context(session_messages)
                messages.append({"type": "info", "text": "Using conversation history for context-aware response..."})

            enhanced_query = query
            if conversation_context:
                enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
            docs = rag_system.similarity_search(enhanced_query, k=3)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                response = f"Based on the documents:\n\n{context[:500]}..."
                
                # Convert docs to source chunks format for document viewer
                source_chunks = []
                for i, doc in enumerate(docs):
                    chunk = {
                        'text': doc.page_content,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'page': doc.metadata.get('page'),
                        'paragraph': doc.metadata.get('paragraph', i),
                        'score': 1.0  # Default score for Basic RAG
                    }
                    source_chunks.append(chunk)
                
                return response, context, source_chunks, messages
            else:
                return "No relevant documents found.", "", None, messages

        elif technique == "Explainable Retrieval":
            try:
                messages.append({"type": "info", "text": "🔄 Running Explainable Retrieval..."})
                messages.append({"type": "info", "text": "**Explainable Retrieval Process:**"})
                messages.append({"type": "info", "text": "1. Retrieving relevant document chunks..."})
                messages.append({"type": "info", "text": "2. Generating explanations for each retrieved chunk..."})
                messages.append({"type": "info", "text": "3. Synthesizing a comprehensive answer with reasoning..."})

                conversation_context = ""
                if len(session_messages) > 1:
                    conversation_context = build_conversation_context(session_messages)
                    messages.append({"type": "info", "text": "4. Considering conversation history for context-aware response..."})

                enhanced_query = query
                if conversation_context:
                    enhanced_query = f"""
Previous conversation:
{conversation_context}

Current question: {query}

Please answer the current question considering the conversation history above.
"""
                detailed_results = rag_system.run(enhanced_query)
                context = ""
                source_chunks = []
                
                if detailed_results:
                    context = "\n".join([result['content'] for result in detailed_results])
                    
                    # Convert detailed_results to source chunks format for document viewer
                    for i, result in enumerate(detailed_results):
                        chunk = {
                            'text': result['content'],
                            'source': result.get('source', 'Unknown'),
                            'page': result.get('page'),
                            'paragraph': result.get('paragraph', i),
                            'score': result.get('score', 1.0)
                        }
                        source_chunks.append(chunk)

                answer = rag_system.answer(enhanced_query)
                messages.append({"type": "success", "text": "✅ Explainable Retrieval completed"})

                # Add detailed explanations as info messages
                if detailed_results:
                    for i, result in enumerate(detailed_results, 1):
                        messages.append({"type": "info", "text": f"📄 Retrieved Section {i}:\nContent: {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}\n💡 Explanation: {result['explanation']}"})
                else:
                    messages.append({"type": "info", "text": "No detailed explanations available."})

                return answer, context, source_chunks, messages

            except Exception as er_error:
                error_msg = str(er_error)
                messages.append({"type": "error", "text": f"❌ Explainable Retrieval Error: {error_msg}"})
                if "API" in error_msg or "google" in error_msg.lower():
                    messages.append({"type": "warning", "text": "⚠️ This appears to be a Google API issue. Check your internet connection and API key."})
                elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    messages.append({"type": "warning", "text": "⚠️ API rate limit reached. Please wait a moment and try again."})
                return f"Explainable Retrieval failed with error: {error_msg}", "", None, messages

        return "Unknown technique.", "", None, messages

    except Exception as e:
        return f"Error generating response: {str(e)}", "", None, [{"type": "error", "text": str(e)}]
    
def add_message(
    session: dict,
    role: str,
    content: str,
    technique: str = None,
    query_id: str = None,
    source_chunks: list = None
):
    """Add message to session dict and save to database (FastAPI version)"""
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "technique": technique,
        "query_id": query_id,
        "timestamp": datetime.now().isoformat()
    }
    session['messages'].append(message)

    # Store source chunks if provided (for CRAG responses)
    if source_chunks and role == "assistant":
        session['last_source_chunks'][message["id"]] = source_chunks

    # Save to persistent storage (database)
    save_chat_message(
        session['session_id'],
        message["id"],
        role,
        content,
        technique,
        query_id
    )

def get_source_documents_ui(message_id: str, source_chunks: List[Dict]):
    """
    Prepare source document info for frontend UI rendering (FastAPI version).
    Returns a list of document info dicts for the frontend to display.
    """
    if not source_chunks:
        return []

    docs_with_chunks = {}
    for chunk in source_chunks:
        doc_path = chunk.get('source', 'Unknown')
        if doc_path not in docs_with_chunks:
            docs_with_chunks[doc_path] = []
        docs_with_chunks[doc_path].append(chunk)

    documents_ui = []
    for doc_path, chunks in docs_with_chunks.items():
        doc_name = os.path.basename(doc_path) if doc_path != 'Unknown' else 'Uploaded Document'
        avg_score = sum(chunk.get('score', 0) for chunk in chunks) / len(chunks)
        
        # Prepare chunk data with enhanced information
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "id": f"chunk_{message_id}_{i}",
                "content": chunk.get('content', chunk.get('text', chunk.get('page_content', ''))),
                "score": round(chunk.get('score', 0.0), 2),
                "page": chunk.get('page', 1),
                "metadata": chunk.get('metadata', {}),
                "highlighted": True  # Flag to indicate this chunk should be highlighted
            }
            chunk_data.append(chunk_info)
        
        document_id = f"doc_{abs(hash(doc_path))}"
        document_info = {
            "doc_path": doc_path,
            "doc_name": doc_name,
            "chunk_count": len(chunks),
            "avg_score": round(avg_score, 2),
            "chunks": chunk_data,
            "document_id": document_id,
            "message_id": message_id,
            "view_url": f"/document/{document_id}?message_id={message_id}",
            "download_url": f"/document/{document_id}/download",
            "actions": [
                {
                    "type": "link",
                    "label": "📄 View with Highlights",
                    "action": "view_document",
                    "url": f"/document/{document_id}?message_id={message_id}",
                    "key": f"view_{message_id}_{document_id}",
                    "enabled": doc_path != 'Unknown',
                    "target": "_blank"
                },
                {
                    "type": "link",
                    "label": "📥 Download Original",
                    "action": "download_document", 
                    "url": f"/document/{document_id}/download",
                    "key": f"download_{message_id}_{document_id}",
                    "enabled": doc_path != 'Unknown',
                    "target": "_blank"
                }
            ]
        }
        documents_ui.append(document_info)
    
    return documents_ui
def get_message_ui(message: Dict[str, Any], message_index: int, session: dict) -> Dict[str, Any]:
    """
    Prepare a message dict for frontend UI rendering (FastAPI version).
    Includes delete actions, feedback prompts, and source document info.
    """
    ui_message = {
        "id": message["id"],
        "role": message["role"],
        "content": message["content"],
        "timestamp": message["timestamp"],
        "technique": message.get("technique"),
        "actions": [],
        "feedback": None,
        "source_documents_ui": None
    }

    # User message: add delete action
    if message["role"] == "user":
        delete_action = {
            "type": "button",
            "label": "✕",
            "action": "delete",
            "key": f"delete_user_{message['id']}",
            "help": "Delete this question and response"
        }
        ui_message["actions"].append(delete_action)

        # Find corresponding assistant message for delete pair
        assistant_message = None
        messages = session["messages"]
        if message_index is not None and message_index + 1 < len(messages):
            next_message = messages[message_index + 1]
            if next_message["role"] == "assistant":
                assistant_message = next_message
        if assistant_message:
            ui_message["delete_pair"] = {
                "user_message_id": message["id"],
                "assistant_message_id": assistant_message["id"],
                "query_id": assistant_message.get("query_id")
            }
        else:
            ui_message["delete_pair"] = {
                "user_message_id": message["id"],
                "assistant_message_id": None,
                "query_id": None
            }

    # Assistant message: add technique badge
    if message["role"] == "assistant":
        ui_message["technique_badge"] = message.get("technique")

        # Feedback collection prompt if pending
        query_id = message.get("query_id")
        if query_id and query_id in session.get("pending_feedback", {}):
            ui_message["feedback"] = {
                "query_id": query_id,
                "message_id": message["id"],
                "pending": True
            }

        # Source documents for CRAG responses
        if message.get("technique") == "CRAG" and message["id"] in session.get("last_source_chunks", {}):
            source_chunks = session["last_source_chunks"][message["id"]]
            if source_chunks:
                ui_message["source_documents_ui"] = get_source_documents_ui(message["id"], source_chunks)

    return ui_message

def get_document_hash(document_paths):
    """Create a hash of the current document list to detect changes"""
    if not document_paths:
        return None
    # Sort paths and create a simple hash
    sorted_paths = sorted(document_paths)
    return hash(tuple(sorted_paths))

def should_reload_rag_system(session: dict, technique: str, document_paths: list):
    """Check if RAG system should be reloaded due to document changes (FastAPI version)"""
    current_hash = get_document_hash(document_paths)
    
    # If documents changed, clear all cached systems
    if current_hash != session.get('last_document_hash'):
        session['rag_systems'] = {}
        session['last_document_hash'] = current_hash
        return True
    
    # If system not loaded for this technique, need to load
    return technique not in session.get('rag_systems', {})

def build_conversation_context(messages, max_turns=3):
    """
    Build a context string from the last N conversation turns.
    This allows the chatbot to remember and reference previous exchanges.
    
    Args:
        messages: List of message dictionaries from session state
        max_turns: Maximum number of conversation turns to include
        
    Returns:
        String with formatted conversation history
    """
    # Only use recent messages (limited by max_turns)
    recent_messages = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
    
    # Format the conversation
    conversation_lines = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_lines.append(f"{role}: {msg['content']}")
    
    # Join into a single string with line breaks
    conversation_context = "\n".join(conversation_lines)
    return conversation_context

def create_multi_document_basic_rag(document_paths: List[str], chunk_size=1000, chunk_overlap=200):
    """
    Create a Basic RAG system that can handle multiple documents by combining them into a single vectorstore.
    Returns (vectorstore, messages) for frontend UI display.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    messages = []
    vectorstore = None

    try:
        all_documents = []
        processed_files = []

        # Load and process each document
        for file_path in document_paths:
            try:
                content = load_document_content(file_path)
                doc_name = os.path.basename(file_path)
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path, "filename": doc_name}
                )
                all_documents.append(doc)
                processed_files.append(doc_name)
            except Exception as e:
                messages.append({"type": "warning", "text": f"⚠️ Could not process {os.path.basename(file_path)}: {str(e)}"})
                continue

        if not all_documents:
            messages.append({"type": "error", "text": "No documents could be processed successfully."})
            return None, messages

        messages.append({"type": "success", "text": f"✅ Loaded {len(processed_files)} documents: {', '.join(processed_files)}"})

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_documents(all_documents)

        # Clean the texts (remove tab characters)
        cleaned_texts = replace_t_with_space(texts)

        # Create embeddings and vector store using local embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

        messages.append({"type": "info", "text": f"Created vectorstore with {len(cleaned_texts)} chunks from {len(processed_files)} documents"})

        return vectorstore, messages

    except Exception as e:
        messages.append({"type": "error", "text": f"Error creating multi-document Basic RAG: {str(e)}"})
        return None, messages
# === SESSION MANAGEMENT ===

def get_session(session_id: str) -> SessionData:
    """Get or create a session"""
    if session_id not in sessions:
        sessions[session_id] = SessionData(
            session_id=session_id,
            messages=load_chat_history(session_id),
            uploaded_documents=[],
            selected_technique="Adaptive RAG",
            crag_web_search_enabled=True,
            pending_feedback={},
            last_source_chunks={}
        )
    return sessions[session_id]

def create_new_chat_session():
    """Create a new chat session (FastAPI version)"""
    new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    # Initialize session data as SessionData object
    sessions[new_session_id] = SessionData(
        session_id=new_session_id,
        messages=[],
        uploaded_documents=[],
        selected_technique="Adaptive RAG",
        crag_web_search_enabled=True,
        pending_feedback={},
        last_source_chunks={}
    )
    return new_session_id

def switch_to_chat_session(sessions: dict, session_id: str) -> bool:
    """Switch to a specific chat session (FastAPI version)"""
    if session_id not in sessions:
        # If session does not exist, create and initialize it
        sessions[session_id] = SessionData(
            session_id=session_id,
            messages=load_chat_history(session_id),
            uploaded_documents=[],
            selected_technique="Adaptive RAG",
            crag_web_search_enabled=True,
            pending_feedback={},
            last_source_chunks={}
        )
        return True
    return False

# === FASTAPI ROUTES ===

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    # Create a new session ID for new visitors
    session_id = create_new_chat_session()
    session_data = get_session(session_id)
    all_sessions = get_all_chat_sessions()
    
    # Technique descriptions
    technique_descriptions = {
        "Adaptive RAG": "Dynamically adapts retrieval strategy based on query type (Factual, Analytical, Opinion, Contextual)",
        "CRAG": "Corrective RAG that evaluates retrieved documents and falls back to web search if needed",
        "Document Augmentation": "Enhances documents with generated questions for better retrieval",
        "Basic RAG": "Standard similarity-based retrieval and response generation",
        "Explainable Retrieval": "Provides explanations for why each retrieved document chunk is relevant to your query using Gemini AI"
    }
    
    return templates.TemplateResponse("index_fastapi.html", {
        "request": request,
        "session_id": session_id,
        "messages": session_data.messages,
        "uploaded_documents": session_data.uploaded_documents,
        "selected_technique": session_data.selected_technique,
        "crag_web_search_enabled": session_data.crag_web_search_enabled,
        "all_sessions": all_sessions,
        "technique_descriptions": technique_descriptions,
        "rag_techniques": ["Adaptive RAG", "CRAG", "Document Augmentation", "Basic RAG", "Explainable Retrieval"]
    })

@app.post("/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...), session_id: str = Form(...)):
    """Handle file uploads"""
    session_data = get_session(session_id)
    uploaded_paths = []
    
    for file in files:
        if file.filename:
            file_path = save_uploaded_file(file)
            if file_path:
                uploaded_paths.append(file_path)
    
    session_data.uploaded_documents = uploaded_paths
    sessions[session_id] = session_data
    
    return JSONResponse({
        "success": True,
        "message": f"Uploaded {len(uploaded_paths)} document(s)",
        "files": [os.path.basename(path) for path in uploaded_paths]
    })

@app.post("/query")
async def process_query(query_request: QueryRequest):
    """Process a user query"""
    session_data = get_session(query_request.session_id)
    
    # Add user message
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": query_request.query,
        "technique": None,
        "query_id": None,
        "timestamp": datetime.now().isoformat()
    }
    session_data.messages.append(user_message)
    
    # Save user message to database
    save_chat_message(
        query_request.session_id,
        user_message["id"],
        "user",
        query_request.query
    )
    
    # Load RAG system and collect UI messages
    rag_system, messages = load_rag_system(
        query_request.technique,
        session_data.uploaded_documents,
        query_request.crag_web_search_enabled
    )
    if rag_system:
        rag_systems[query_request.technique] = rag_system
    else:
        return JSONResponse({
            "success": False,
            "error": f"Failed to load {query_request.technique}",
            "messages": messages
        })
    # Generate response
    start_time = time.time()
    response, context, source_chunks, rag_messages = get_rag_response(
        query_request.technique,
        query_request.query,
        rag_system,
        session_data.messages
    )
    end_time = time.time()
    response_time = end_time - start_time
    
    # Add assistant message
    assistant_message = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": response,
        "technique": query_request.technique,
        "query_id": None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Store evaluation data
    try:
        document_sources = [os.path.basename(doc) for doc in session_data.uploaded_documents]
        query_id = evaluation_manager.evaluate_rag_response(
            query=query_request.query,
            response=response,
            context=context,
            technique=query_request.technique,
            processing_time=response_time,
            document_sources=document_sources,
            session_id=query_request.session_id
        )
        assistant_message["query_id"] = query_id
        session_data.pending_feedback[query_id] = True
    except Exception as e:
        print(f"Evaluation storage failed: {e}")
    
    session_data.messages.append(assistant_message)
    
    # Store source chunks if available
    if source_chunks:
        session_data.last_source_chunks[assistant_message["id"]] = source_chunks
    
    # Save assistant message to database
    save_chat_message(
        query_request.session_id,
        assistant_message["id"],
        "assistant",
        response,
        query_request.technique,
        assistant_message.get("query_id")
    )
    source_documents_ui= get_source_documents_ui(assistant_message["id"], source_chunks)
    sessions[query_request.session_id] = session_data
    all_messages = messages + rag_messages
    ui_messages = [
        get_message_ui(msg, idx, {
            "messages": session_data.messages, 
            "pending_feedback": session_data.pending_feedback, 
            "last_source_chunks": session_data.last_source_chunks
        })
        for idx, msg in enumerate(session_data.messages)
    ]
    return JSONResponse({
        "success": True,
        "response": response,
        "technique": query_request.technique,
        "query_id": assistant_message.get("query_id"),
        "source_chunks": source_chunks,
        "processing_time": response_time,
        "messages": all_messages,
        "source_documents_ui": source_documents_ui,
        "ui_messages": ui_messages
    })


@app.post("/feedback")
async def submit_feedback(feedback_request: FeedbackRequest):
    """Submit user feedback"""
    messages = []
    try:
        feedback = UserFeedback(
            helpfulness=feedback_request.helpfulness,
            accuracy=feedback_request.accuracy,
            clarity=feedback_request.clarity,
            overall_rating=feedback_request.overall_rating,
            comments=feedback_request.comments,
            timestamp=datetime.now().isoformat()
        )
        
        evaluation_manager.add_user_feedback(feedback_request.query_id, feedback)
        messages.append({"type": "success", "text": "Thank you for your feedback! 🙏"})
        
        # Optionally, remove from pending feedback in session (if you track it)
        # session_data = get_session_by_query_id(feedback_request.query_id)
        # if session_data and feedback_request.query_id in session_data.pending_feedback:
        #     del session_data.pending_feedback[feedback_request.query_id]
        
        return JSONResponse({
            "success": True,
            "messages": messages
        })
    except Exception as e:
        messages.append({"type": "error", "text": f"Error submitting feedback: {str(e)}"})
        return JSONResponse({
            "success": False,
            "messages": messages
        })

@app.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    return JSONResponse(get_all_chat_sessions())

@app.post("/sessions/new")
async def create_new_session():
    """Create a new chat session"""
    session_id = create_new_chat_session()
    session_data = sessions[session_id]
    return JSONResponse({
        "success": True,
        "session_id": session_id,
        "messages": session_data.messages
    })

@app.post("/sessions/switch")
async def switch_session(request: Request):
    """Switch to a different session"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        if not session_id:
            return JSONResponse({
                "success": False,
                "message": "Missing session_id"
            }, status_code=400)
        
        switched = switch_to_chat_session(sessions, session_id)
        session_data = sessions[session_id]
        return JSONResponse({
            "success": True,
            "switched": switched,
            "session_id": session_id,
            "messages": session_data.messages
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error switching session: {str(e)}"
        }, status_code=500)
    
@app.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, request: Request):
    """Rename a chat session"""
    try:
        data = await request.json()
        new_title = data.get("title")
        
        if not new_title:
            return JSONResponse({
                "success": False,
                "message": "Title is required"
            }, status_code=400)
        
        success = rename_chat_session(session_id, new_title)
        
        return JSONResponse({
            "success": success,
            "message": "Session renamed successfully" if success else "Failed to rename session"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error renaming session: {str(e)}"
        }, status_code=500)

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session (FastAPI version)"""
    success, new_session_id = delete_chat_session(session_id)
    response = {"success": success}
    if new_session_id:
        response["new_session_id"] = new_session_id
        response["message"] = "Session deleted. New session created."
    else:
        response["message"] = "Session deleted."
    return JSONResponse(response)

@app.get("/analytics")
async def get_analytics():
    """Get analytics data"""
    try:
        # This would need to be adapted from the Streamlit analytics dashboard
        return JSONResponse({
            "success": True,
            "data": "Analytics data would go here"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@app.get("/document/{document_id}")
async def view_document(request: Request, document_id: str, message_id: str = None):
    """Serve the PDF viewer page with highlighting information"""
    try:
        # Find the file path for the document_id
        file_path = get_document_path_from_id(document_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get the relevant chunks for highlighting
        chunks = get_chunks_for_message(message_id) if message_id else []
        
        # Debug: Print chunk structure
        print(f"Debug: Found {len(chunks)} chunks for message {message_id}")
        if chunks:
            print(f"Debug: First chunk structure: {type(chunks[0])}")
            print(f"Debug: First chunk keys: {chunks[0].keys() if isinstance(chunks[0], dict) else 'Not a dict'}")
            print(f"Debug: First chunk sample: {str(chunks[0])[:200]}...")
        
        # Process chunks to extract page numbers and text
        processed_chunks = []
        for chunk in chunks:
            # Handle different chunk data structures
            if isinstance(chunk, dict):
                text_content = (
                    chunk.get('content') or 
                    chunk.get('text') or 
                    chunk.get('page_content') or 
                    str(chunk.get('source', ''))
                )
                processed_chunk = {
                    "text": text_content,
                    "page": chunk.get('page', 1),
                    "score": chunk.get('score', 0.0),
                    "metadata": chunk.get('metadata', {})
                }
            else:
                # Handle non-dict chunks (fallback)
                text_content = str(chunk) if chunk else ""
                processed_chunk = {
                    "text": text_content,
                    "page": 1,
                    "score": 0.0,
                    "metadata": {}
                }
            processed_chunks.append(processed_chunk)
        
        return templates.TemplateResponse("pdf_viewer.html", {
            "request": request,
            "pdf_url": f"/document/{document_id}/download",
            "chunks": processed_chunks,
            "doc_name": os.path.basename(file_path),
            "document_id": document_id,
            "message_id": message_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

@app.get("/document/{document_id}/download")
async def download_document(document_id: str):
    """Serve the original document file"""
    try:
        file_path = get_document_path_from_id(document_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Determine the media type
        media_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        
        return FileResponse(
            file_path, 
            media_type=media_type, 
            filename=os.path.basename(file_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving document: {str(e)}")

@app.get("/document/{document_id}/content")
async def get_document_content(document_id: str, page: int = None):
    """Get document content for highlighting (JSON API)"""
    try:
        file_path = get_document_path_from_id(document_id)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        if page is not None:
            # Get specific page content
            text_content = extract_text_from_pdf_page(file_path, page)
            return JSONResponse({
                "page": page,
                "content": text_content
            })
        else:
            # Get full document content
            from src.document_augmentation import load_document_content
            content = load_document_content(file_path)
            return JSONResponse({
                "content": content
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document content: {str(e)}")

# === DOCUMENT SERVING HELPER FUNCTIONS ===

def get_document_path_from_id(document_id: str) -> str:
    """Map document_id back to file path by searching all sessions"""
    for session_id, session_data in sessions.items():
        # Check uploaded_documents (default)
        if hasattr(session_data, 'uploaded_documents'):
            documents = session_data.uploaded_documents
        else:
            documents = session_data.get('uploaded_documents', [])

        for doc_path in documents:
            if f"doc_{abs(hash(doc_path))}" == document_id:
                return doc_path

        # Special handling for CRAG sessions
        # If the session has a crag_document_path attribute, check it
        crag_doc_path = getattr(session_data, 'crag_document_path', None)
        if crag_doc_path and f"doc_{abs(hash(crag_doc_path))}" == document_id:
            return crag_doc_path

        # Or if stored in dict
        crag_doc_path_dict = session_data.get('crag_document_path', None) if isinstance(session_data, dict) else None
        if crag_doc_path_dict and f"doc_{abs(hash(crag_doc_path_dict))}" == document_id:
            return crag_doc_path_dict

    return None

def get_chunks_for_message(message_id: str) -> List[Dict]:
    """Get chunks associated with a specific message for highlighting"""
    if not message_id:
        return []
    
    # Search all sessions for the message_id
    for session_id, session_data in sessions.items():
        if hasattr(session_data, 'last_source_chunks'):
            source_chunks = session_data.last_source_chunks
        else:
            source_chunks = session_data.get('last_source_chunks', {})
        
        if message_id in source_chunks:
            chunks = source_chunks[message_id]
            # Ensure chunks are properly formatted dictionaries
            if chunks and isinstance(chunks, list):
                formatted_chunks = []
                for chunk in chunks:
                    if isinstance(chunk, dict):
                        # Ensure required fields exist
                        formatted_chunk = {
                            'text': chunk.get('text', chunk.get('content', '')),
                            'source': chunk.get('source', 'Unknown'),
                            'page': chunk.get('page'),
                            'paragraph': chunk.get('paragraph'),
                            'score': chunk.get('score', 1.0)
                        }
                        formatted_chunks.append(formatted_chunk)
                    else:
                        # Handle case where chunk is not a dict (e.g., string or other object)
                        formatted_chunk = {
                            'text': str(chunk),
                            'source': 'Unknown',
                            'page': None,
                            'paragraph': None,
                            'score': 1.0
                        }
                        formatted_chunks.append(formatted_chunk)
                return formatted_chunks
    
    return []

def extract_text_from_pdf_page(file_path: str, page_num: int) -> str:
    """Extract text from a specific PDF page"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        if page_num <= len(reader.pages):
            page = reader.pages[page_num - 1]  # 0-indexed
            return page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF page {page_num}: {e}")
    return ""
