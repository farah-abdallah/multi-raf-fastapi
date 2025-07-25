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

from src.crag import CRAG
from src.document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings, load_document_content
from src.utils.helpers import encode_document, replace_t_with_space

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
    """Load only the CRAG RAG system with error handling and collect UI messages."""
    messages = []
    rag_system = None
    try:
        messages.append({"type": "info", "text": f"Loading CRAG..."})
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
    except Exception as e:
        messages.append({"type": "error", "text": f"Error loading CRAG: {str(e)}"})
    return rag_system, messages
def get_rag_response(technique: str, query: str, rag_system, session_messages: list):
    """
    Get response from the CRAG RAG system and return response, context, source_chunks, and UI messages.
    """
    messages = []
    try:
        context = ""  # Will store retrieved context for evaluation
        source_chunks = None  # For CRAG responses

        # Only CRAG logic
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

    except Exception as e:
        return f"Error generating response: {str(e)}", "", None, [{"type": "error", "text": str(e)}]

def add_message(
    session: dict,
    role: str,
    content: str,
    query_id: str = None,
    source_chunks: list = None
):
    """Add message to session dict and save to database (FastAPI version, CRAG only)"""
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
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
        query_id=query_id
    )


# In your get_source_documents_ui function in chatbot_app.py
# Update your get_source_documents_ui function in chatbot_app.py
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
                "highlighted": True
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
            "view_highlighted_url": f"/document/{document_id}/view-highlighted?message_id={message_id}",
            "download_url": f"/document/{document_id}/download",
            "download_highlighted_url": f"/document/{document_id}/download-highlighted?message_id={message_id}",
            "actions": [
                {
                    "type": "link",
                    "label": "📄 View PDF with Highlights",
                    "action": "view_document",
                    "url": f"/document/{document_id}?message_id={message_id}",
                    "key": f"view_{message_id}_{document_id}",
                    "enabled": doc_path != 'Unknown',
                    "target": "_blank"
                },
                {
                    "type": "link",
                    "label": "🎯 View Highlighted PDF",
                    "action": "view_highlighted",
                    "url": f"/document/{document_id}/view-highlighted?message_id={message_id}",
                    "key": f"view_highlighted_{message_id}_{document_id}",
                    "enabled": doc_path != 'Unknown',
                    "target": "_blank"
                },
                {
                    "type": "link",
                    "label": "📥 Download Original PDF",
                    "action": "download_document", 
                    "url": f"/document/{document_id}/download",
                    "key": f"download_{message_id}_{document_id}",
                    "enabled": doc_path != 'Unknown',
                    "target": "_blank"
                },
                {
                    "type": "link",
                    "label": "💾 Download with Highlights",
                    "action": "download_highlighted", 
                    "url": f"/document/{document_id}/download-highlighted?message_id={message_id}",
                    "key": f"download_highlighted_{message_id}_{document_id}",
                    "enabled": doc_path != 'Unknown',
                    "target": "_blank"
                }
            ]
        }
        documents_ui.append(document_info)
    
    return documents_ui

def get_message_ui(message: Dict[str, Any], message_index: int, session: dict) -> Dict[str, Any]:
    """
    Prepare a message dict for frontend UI rendering (FastAPI version, CRAG only).
    Includes delete actions, feedback prompts, and source document info.
    """
    ui_message = {
        "id": message["id"],
        "role": message["role"],
        "content": message["content"],
        "timestamp": message["timestamp"],
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

    # Assistant message: CRAG logic only
    if message["role"] == "assistant":
        # Feedback collection prompt if pending
        query_id = message.get("query_id")
        if query_id and query_id in session.get("pending_feedback", {}):
            ui_message["feedback"] = {
                "query_id": query_id,
                "message_id": message["id"],
                "pending": True
            }

        # Source documents for CRAG responses
        if message["id"] in session.get("last_source_chunks", {}):
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

def should_reload_rag_system(session: dict, document_paths: list):
    """Check if CRAG system should be reloaded due to document changes (FastAPI version)"""
    current_hash = get_document_hash(document_paths)
    
    # If documents changed, clear cached CRAG system
    if current_hash != session.get('last_document_hash'):
        session['rag_systems'] = {}
        session['last_document_hash'] = current_hash
        return True
    
    # If CRAG system not loaded, need to load
    return "CRAG" not in session.get('rag_systems', {})

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


# === SESSION MANAGEMENT ===

def get_session(session_id: str) -> SessionData:
    """Get or create a session (CRAG only)"""
    if session_id not in sessions:
        sessions[session_id] = SessionData(
            session_id=session_id,
            messages=load_chat_history(session_id),
            uploaded_documents=[],
            crag_web_search_enabled=True,
            pending_feedback={},
            last_source_chunks={}
        )
    return sessions[session_id]

def create_new_chat_session():
    """Create a new chat session (FastAPI version, CRAG only)"""
    new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    # Initialize session data as SessionData object
    sessions[new_session_id] = SessionData(
        session_id=new_session_id,
        messages=[],
        uploaded_documents=[],
        crag_web_search_enabled=True,
        pending_feedback={},
        last_source_chunks={}
    )
    return new_session_id

def switch_to_chat_session(sessions: dict, session_id: str) -> bool:
    """Switch to a specific chat session (FastAPI version, CRAG only)"""
    if session_id not in sessions:
        # If session does not exist, create and initialize it
        sessions[session_id] = SessionData(
            session_id=session_id,
            messages=load_chat_history(session_id),
            uploaded_documents=[],
            crag_web_search_enabled=True,
            pending_feedback={},
            last_source_chunks={}
        )
        return True
    return False

# === FASTAPI ROUTES ===

# Add these routes to your chatbot_app.py

@app.get("/document/{document_id}/download-highlighted")
async def download_highlighted_pdf(document_id: str, message_id: str = None):
    """Download PDF with highlighted chunks"""
    try:
        # Get the document path and chunks
        doc_path = get_document_path_by_id(document_id)
        if not doc_path or not os.path.exists(doc_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks for highlighting if message_id provided
        chunks = []
        if message_id:
            chunks = get_chunks_for_message(message_id)
        
        # Create highlighted PDF
        highlighted_pdf_path = create_highlighted_pdf(doc_path, chunks, message_id)
        
        if not highlighted_pdf_path or not os.path.exists(highlighted_pdf_path):
            # Fallback to original PDF if highlighting fails
            return FileResponse(
                doc_path,
                media_type='application/pdf',
                filename=f"{os.path.basename(doc_path)}"
            )
        
        return FileResponse(
            highlighted_pdf_path,
            media_type='application/pdf',
            filename=f"highlighted_{os.path.basename(doc_path)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{document_id}/download")
async def download_original_pdf(document_id: str):
    """Download original PDF without highlights"""
    try:
        doc_path = get_document_path_by_id(document_id)
        if not doc_path or not os.path.exists(doc_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        return FileResponse(
            doc_path,
            media_type='application/pdf',
            filename=os.path.basename(doc_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_document_path_by_id(document_id: str) -> str:
    """Get document path from document ID"""
    # This function should map document IDs back to file paths
    # You may need to store this mapping in your session or database
    for session_data in sessions.values():
        for doc_path in session_data.uploaded_documents:
            if f"doc_{abs(hash(doc_path))}" == document_id:
                return doc_path
    return None

def create_highlighted_pdf(original_pdf_path: str, chunks: List[Dict], message_id: str) -> str:
    """Create a PDF with highlighted chunks"""
    try:
        import fitz  # PyMuPDF
        from datetime import datetime
        
        # Open the original PDF
        doc = fitz.open(original_pdf_path)
        
        # Create highlights for each chunk
        for chunk in chunks:
            page_num = chunk.get('page', 1) - 1  # PyMuPDF uses 0-based indexing
            if page_num < 0 or page_num >= len(doc):
                continue
                
            page = doc[page_num]
            chunk_text = chunk.get('content', chunk.get('text', ''))
            
            if not chunk_text:
                continue
            
            # Search for text and highlight
            text_instances = page.search_for(chunk_text[:100])  # Search first 100 chars
            if not text_instances:
                # Try with first few words if full text not found
                words = chunk_text.split()[:10]
                search_text = ' '.join(words)
                text_instances = page.search_for(search_text)
            
            # Add highlight annotations
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors({"stroke": [1, 1, 0]})  # Yellow highlight
                highlight.update()
        
        # Save the highlighted PDF
        output_dir = "temp_highlighted_pdfs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"highlighted_{message_id}_{timestamp}.pdf")
        
        doc.save(output_path)
        doc.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error creating highlighted PDF: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface (CRAG only)"""
    # Create a new session ID for new visitors
    session_id = create_new_chat_session()
    session_data = get_session(session_id)
    all_sessions = get_all_chat_sessions()
    
    # CRAG technique description only
    crag_description = "Corrective RAG that evaluates retrieved documents and falls back to web search if needed"
    
    return templates.TemplateResponse("index_fastapi.html", {
        "request": request,
        "session_id": session_id,
        "messages": session_data.messages,
        "uploaded_documents": session_data.uploaded_documents,
        "crag_web_search_enabled": session_data.crag_web_search_enabled,
        "all_sessions": all_sessions,
        "crag_description": crag_description
    })


# Add this route to your chatbot_app.py

@app.get("/document/{document_id}/view-highlighted")
async def view_highlighted_pdf(document_id: str, message_id: str = None):
    """View highlighted PDF directly in browser"""
    try:
        # Get the document path and chunks
        doc_path = get_document_path_by_id(document_id)
        if not doc_path or not os.path.exists(doc_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks for highlighting if message_id provided
        chunks = []
        if message_id:
            chunks = get_chunks_for_message(message_id)
        
        # Create highlighted PDF
        highlighted_pdf_path = create_highlighted_pdf(doc_path, chunks, message_id)
        
        if not highlighted_pdf_path or not os.path.exists(highlighted_pdf_path):
            # Fallback to original PDF if highlighting fails
            return FileResponse(
                doc_path,
                media_type='application/pdf',
                filename=f"{os.path.basename(doc_path)}"
            )
        
        # Return the highlighted PDF for inline viewing (not download)
        return FileResponse(
            highlighted_pdf_path,
            media_type='application/pdf',
            headers={"Content-Disposition": "inline"}  # This makes it open in browser instead of download
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    """Process a user query (CRAG only)"""
    session_data = get_session(query_request.session_id)
    
    # Add user message
    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": query_request.query,
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
    
    # Load CRAG system and collect UI messages
    rag_system, messages = load_rag_system(
        "CRAG",
        session_data.uploaded_documents,
        query_request.crag_web_search_enabled
    )
    if rag_system:
        rag_systems["CRAG"] = rag_system
    else:
        return JSONResponse({
            "success": False,
            "error": "Failed to load CRAG",
            "messages": messages
        })
    # Generate response
    start_time = time.time()
    response, context, source_chunks, rag_messages = get_rag_response(
        "CRAG",
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
        "timestamp": datetime.now().isoformat()
    }
    
    # Store evaluation data
    try:
        document_sources = [os.path.basename(doc) for doc in session_data.uploaded_documents]
        query_id = evaluation_manager.evaluate_rag_response(
            query=query_request.query,
            response=response,
            context=context,
            technique="CRAG",
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
        query_id=assistant_message.get("query_id")
    )
    source_documents_ui = get_source_documents_ui(assistant_message["id"], source_chunks)
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
