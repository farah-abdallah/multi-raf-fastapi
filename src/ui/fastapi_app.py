"""
FastAPI Multi-RAG Chatbot Application with Document Upload, Message History, and Comprehensive Evaluation

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

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Add the project root to the path
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

# === PYDANTIC MODELS ===

class ChatMessage(BaseModel):
    query: str
    technique: str
    crag_web_search_enabled: bool = True
    session_id: Optional[str] = None

class FeedbackSubmission(BaseModel):
    query_id: str
    helpfulness: int
    accuracy: int
    clarity: int
    overall_rating: int
    comments: Optional[str] = ""

class SessionSwitch(BaseModel):
    session_id: str

class SessionRename(BaseModel):
    session_id: str
    new_title: str

# === GLOBAL VARIABLES ===
app = FastAPI(title="Multi-RAG Chatbot", description="FastAPI implementation of Multi-RAG Chatbot")

# In-memory session storage (in production, use Redis or similar)
sessions_data = {}
rag_systems = {}
evaluation_manager = None

# Create templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# Create static directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# === DATABASE FUNCTIONS ===

def init_chat_database():
    """Initialize database for persistent chat storage"""
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            message_id TEXT,
            message_type TEXT,
            content TEXT,
            technique TEXT,
            query_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_chat_message(session_id: str, message_id: str, message_type: str, content: str, technique: str = None, query_id: str = None):
    """Save chat message to database"""
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
    """Load chat history from database"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT message_id, message_type, content, technique, query_id, timestamp
            FROM chat_sessions 
            WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            message_id, message_type, content, technique, query_id, timestamp = row
            message = {
                'id': message_id,
                'role': message_type,
                'content': content,
                'technique': technique,
                'query_id': query_id,
                'timestamp': timestamp
            }
            messages.append(message)
        
        conn.close()
        return messages
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

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
    """Delete an entire chat session"""
    try:
        conn = sqlite3.connect('chat_history.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chat_sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return False

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

# === RAG SYSTEM FUNCTIONS ===

def get_or_create_session_id(session_id: str = None):
    """Get or create a session ID"""
    if session_id and session_id in sessions_data:
        return session_id
    
    new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    sessions_data[new_session_id] = {
        'messages': [],
        'uploaded_documents': [],
        'selected_technique': 'Adaptive RAG',
        'crag_web_search_enabled': True,
        'pending_feedback': {},
        'last_source_chunks': {},
        'last_document_hash': None
    }
    return new_session_id

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path"""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.filename)
        
        with open(file_path, "wb") as buffer:
            content = uploaded_file.file.read()
            buffer.write(content)
        
        return file_path
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return None

def get_document_hash(document_paths):
    """Create a hash of the current document list to detect changes"""
    if not document_paths:
        return None
    sorted_paths = sorted(document_paths)
    return hash(tuple(sorted_paths))

def should_reload_rag_system(technique, document_paths, session_id):
    """Check if RAG system should be reloaded due to document changes"""
    current_hash = get_document_hash(document_paths)
    
    # If documents changed, clear all cached systems
    if current_hash != sessions_data[session_id]['last_document_hash']:
        global rag_systems
        rag_systems = {}
        sessions_data[session_id]['last_document_hash'] = current_hash
        return True
    
    # If system not loaded for this technique, need to load
    return technique not in rag_systems

def load_rag_system(technique: str, document_paths: List[str] = None, crag_web_search_enabled: bool = None):
    """Load the specified RAG system with error handling"""
    try:
        if not document_paths:
            return None

        if technique == "Adaptive RAG":
            rag_system = AdaptiveRAG()
            for doc_path in document_paths:
                rag_system.add_document(doc_path)
            return rag_system

        elif technique == "CRAG":
            crag_system = CRAG(web_search_enabled=crag_web_search_enabled)
            for doc_path in document_paths:
                crag_system.add_document(doc_path)
            return crag_system

        elif technique == "Document Augmentation":
            processor = DocumentProcessor()
            for doc_path in document_paths:
                processor.add_document(doc_path)
            return processor

        elif technique == "Basic RAG":
            return create_multi_document_basic_rag(document_paths)

        elif technique == "Explainable Retrieval":
            explainable_rag = ExplainableRAGMethod()
            for doc_path in document_paths:
                explainable_rag.add_document(doc_path)
            return explainable_rag

    except Exception as e:
        print(f"Error loading {technique}: {str(e)}")
        return None

def create_multi_document_basic_rag(document_paths: List[str], chunk_size=1000, chunk_overlap=200):
    """Create a Basic RAG system that can handle multiple documents"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    try:
        all_documents = []
        processed_files = []
        
        # Load and process each document
        for file_path in document_paths:
            try:
                content = load_document_content(file_path)
                doc_name = os.path.basename(file_path)
                
                # Create a document object with metadata
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path, "filename": doc_name}
                )
                all_documents.append(doc)
                processed_files.append(doc_name)
                
            except Exception as e:
                print(f"⚠️ Could not process {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_documents:
            raise ValueError("No documents could be processed successfully")
        
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
        
        return vectorstore
        
    except Exception as e:
        print(f"Error creating multi-document Basic RAG: {str(e)}")
        return None

def get_rag_response(technique: str, query: str, rag_system, messages: List[Dict] = None):
    """Get response from the specified RAG system"""
    try:
        context = ""
        source_chunks = None
        
        if technique == "Adaptive RAG":
            response = rag_system.process_query(query)
            context = str(response.get('context', ''))
            
        elif technique == "CRAG":
            result = rag_system.process_query(query)
            response = result['answer']
            context = result.get('context', '')
            source_chunks = result.get('source_chunks', [])
            
        elif technique == "Document Augmentation":
            conversation_context = ""
            if messages and len(messages) > 1:
                conversation_context = build_conversation_context(messages, max_turns=3)
            
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
            
            docs = rag_system.get_relevant_documents(enhanced_query)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                response = rag_system.generate_answer(enhanced_query, docs)
            else:
                response = "I couldn't find relevant information in the documents to answer your question."
                
        elif technique == "Basic RAG":
            conversation_context = ""
            if messages and len(messages) > 1:
                conversation_context = build_conversation_context(messages, max_turns=3)
            
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
            
            docs = rag_system.similarity_search(enhanced_query, k=3)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                from src.llm.gemini import get_llm_response
                response = get_llm_response(enhanced_query, context)
            else:
                response = "I couldn't find relevant information in the documents to answer your question."
                
        elif technique == "Explainable Retrieval":
            try:
                result = rag_system.process_query_with_explanation(query)
                response = result['answer']
                context = result.get('context', '')
                
            except Exception as er_error:
                response = f"Error in explainable retrieval: {str(er_error)}"
                context = ""
        
        return response, context, source_chunks
        
    except Exception as e:
        return f"Error generating response: {str(e)}", "", None

def build_conversation_context(messages, max_turns=3):
    """Build a context string from the last N conversation turns"""
    recent_messages = messages[-max_turns*2:] if len(messages) > max_turns*2 else messages
    
    conversation_lines = []
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_lines.append(f"{role}: {msg['content']}")
    
    return "\n".join(conversation_lines)

def add_message(session_id: str, role: str, content: str, technique: str = None, query_id: str = None, source_chunks: list = None):
    """Add message to session and save to database"""
    message_id = str(uuid.uuid4())
    message = {
        "id": message_id,
        "role": role,
        "content": content,
        "technique": technique,
        "query_id": query_id,
        "timestamp": datetime.now().isoformat()
    }
    
    sessions_data[session_id]['messages'].append(message)
    
    # Store source chunks if provided
    if source_chunks and role == "assistant":
        sessions_data[session_id]['last_source_chunks'][message_id] = source_chunks
    
    # Save to database
    save_chat_message(session_id, message_id, role, content, technique, query_id)
    
    return message_id

# === INITIALIZATION ===

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global evaluation_manager
    
    # Initialize database
    init_chat_database()
    
    # Initialize evaluation manager
    evaluation_manager = EvaluationManager()

# === API ROUTES ===

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads"""
    uploaded_paths = []
    
    for file in files:
        if file.filename:
            file_path = save_uploaded_file(file)
            if file_path:
                uploaded_paths.append(file_path)
    
    return {"uploaded_files": uploaded_paths, "count": len(uploaded_paths)}

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Handle chat messages"""
    try:
        session_id = get_or_create_session_id(message.session_id)
        
        # Get session data
        session_data = sessions_data[session_id]
        
        # Add user message
        user_message_id = add_message(session_id, "user", message.query)
        
        # Check if RAG system needs reloading
        if should_reload_rag_system(message.technique, session_data['uploaded_documents'], session_id):
            pass  # Will load below
        
        # Load RAG system
        rag_system = load_rag_system(
            message.technique, 
            session_data['uploaded_documents'], 
            message.crag_web_search_enabled
        )
        
        if not rag_system:
            return {"error": "Could not load RAG system. Please upload documents first."}
        
        # Store RAG system for future use
        rag_systems[message.technique] = rag_system
        
        # Generate response
        start_time = time.time()
        response, context, source_chunks = get_rag_response(
            message.technique, 
            message.query, 
            rag_system, 
            session_data['messages']
        )
        end_time = time.time()
        response_time = end_time - start_time
        
        # Add assistant message
        assistant_message_id = add_message(
            session_id, "assistant", response, message.technique, None, source_chunks
        )
        
        # Store evaluation data
        if evaluation_manager:
            try:
                document_sources = [os.path.basename(doc) for doc in session_data['uploaded_documents']]
                query_id = evaluation_manager.evaluate_rag_response(
                    query=message.query,
                    response=response,
                    context=context,
                    technique=message.technique,
                    processing_time=response_time,
                    document_sources=document_sources,
                    session_id=session_id
                )
                
                # Update the assistant message with query_id
                for msg in session_data['messages']:
                    if msg['id'] == assistant_message_id:
                        msg['query_id'] = query_id
                        break
                
                # Mark for feedback
                session_data['pending_feedback'][query_id] = True
                
            except Exception as eval_error:
                print(f"Evaluation storage failed: {eval_error}")
        
        return {
            "response": response,
            "technique": message.technique,
            "session_id": session_id,
            "message_id": assistant_message_id,
            "source_chunks": source_chunks,
            "processing_time": response_time
        }
        
    except Exception as e:
        return {"error": f"Error processing chat: {str(e)}"}

@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str):
    """Get messages for a session"""
    if session_id not in sessions_data:
        # Load from database
        messages = load_chat_history(session_id)
        sessions_data[session_id] = {
            'messages': messages,
            'uploaded_documents': [],
            'selected_technique': 'Adaptive RAG',
            'crag_web_search_enabled': True,
            'pending_feedback': {},
            'last_source_chunks': {},
            'last_document_hash': None
        }
    
    return {"messages": sessions_data[session_id]['messages']}

@app.get("/api/sessions")
async def get_sessions():
    """Get all chat sessions"""
    sessions = get_all_chat_sessions()
    return {"sessions": sessions}

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """Submit user feedback"""
    try:
        if evaluation_manager:
            user_feedback = UserFeedback(
                helpfulness=feedback.helpfulness,
                accuracy=feedback.accuracy,
                clarity=feedback.clarity,
                overall_rating=feedback.overall_rating,
                comments=feedback.comments
            )
            evaluation_manager.add_user_feedback(feedback.query_id, user_feedback)
            
            # Remove from pending feedback for all sessions
            for session_data in sessions_data.values():
                if feedback.query_id in session_data['pending_feedback']:
                    del session_data['pending_feedback'][feedback.query_id]
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        return {"error": f"Error submitting feedback: {str(e)}"}

@app.post("/api/sessions/new")
async def create_new_session():
    """Create a new chat session"""
    session_id = get_or_create_session_id()
    return {"session_id": session_id}

@app.post("/api/sessions/switch")
async def switch_session(session_switch: SessionSwitch):
    """Switch to a different session"""
    session_id = session_switch.session_id
    if session_id not in sessions_data:
        messages = load_chat_history(session_id)
        sessions_data[session_id] = {
            'messages': messages,
            'uploaded_documents': [],
            'selected_technique': 'Adaptive RAG',
            'crag_web_search_enabled': True,
            'pending_feedback': {},
            'last_source_chunks': {},
            'last_document_hash': None
        }
    
    return {"status": "success", "session_id": session_id}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    success = delete_chat_session(session_id)
    if success and session_id in sessions_data:
        del sessions_data[session_id]
    
    return {"status": "success" if success else "error"}

@app.put("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, rename_data: SessionRename):
    """Rename a chat session"""
    success = rename_chat_session(session_id, rename_data.new_title)
    return {"status": "success" if success else "error"}

@app.get("/api/analytics")
async def get_analytics():
    """Get analytics data"""
    try:
        if evaluation_manager:
            # Get analytics data (you'll need to implement this method)
            analytics_data = evaluation_manager.get_analytics_summary()
            return analytics_data
        else:
            return {"error": "Evaluation manager not initialized"}
    except Exception as e:
        return {"error": f"Error getting analytics: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
