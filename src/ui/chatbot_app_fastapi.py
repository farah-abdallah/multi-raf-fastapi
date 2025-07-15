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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

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

# FastAPI app instance
app = FastAPI(title="Multi-RAG Chatbot", description="A comprehensive RAG evaluation platform")

# Templates directory
templates = Jinja2Templates(directory="src/ui/templates")

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
            messages.append({
                "id": message_id,
                "role": role,
                "content": content,
                "technique": technique,
                "query_id": query_id,
                "timestamp": timestamp
            })
        
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

def delete_chat_session(session_id):
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

# === RAG SYSTEM FUNCTIONS (adapted from Streamlit version) ===

def save_uploaded_file(uploaded_file: UploadFile):
    """Save uploaded file to temporary directory and return path"""
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.filename)
        
        with open(file_path, "wb") as buffer:
            content = uploaded_file.file.read()
            buffer.write(content)
        
        return file_path
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return None

def load_rag_system(technique: str, document_paths: List[str] = None, crag_web_search_enabled: bool = None):
    """Load the specified RAG system with error handling"""
    try:
        if not document_paths:
            raise ValueError("No documents uploaded")

        if technique == "Adaptive RAG":
            return AdaptiveRAG(document_paths=document_paths)
        
        elif technique == "CRAG":
            return CRAG(
                document_paths=document_paths, 
                web_search_enabled=crag_web_search_enabled if crag_web_search_enabled is not None else True
            )
        
        elif technique == "Document Augmentation":
            processor = DocumentProcessor()
            embeddings = SentenceTransformerEmbeddings()
            processed_docs = []
            
            for doc_path in document_paths:
                content = load_document_content(doc_path)
                processed_doc = processor.process_document(content, os.path.basename(doc_path))
                processed_docs.append(processed_doc)
            
            return processor
        
        elif technique == "Basic RAG":
            return create_multi_document_basic_rag(document_paths)
        
        elif technique == "Explainable Retrieval":
            return ExplainableRAGMethod(document_paths=document_paths)
        
        else:
            raise ValueError(f"Unknown RAG technique: {technique}")
            
    except Exception as e:
        print(f"Error loading {technique}: {str(e)}")
        return None

def get_rag_response(technique: str, query: str, rag_system, session_messages: List[Dict]):
    """Get response from the specified RAG system and return response, context, and optionally source_chunks"""
    try:
        context = ""  # Will store retrieved context for evaluation
        source_chunks = None  # For CRAG responses
        
        if technique == "Adaptive RAG":
            response = rag_system.answer_query(query)
            context = response  # For Adaptive RAG, the response includes context
        
        elif technique == "CRAG":
            result = rag_system.answer_query(query)
            if isinstance(result, dict):
                response = result.get('answer', str(result))
                context = result.get('context', '')
                source_chunks = result.get('source_chunks', [])
            else:
                response = str(result)
        
        elif technique == "Document Augmentation":
            # Build conversation context from previous messages
            conversation_context = ""
            if len(session_messages) > 1:
                conversation_context = build_conversation_context(session_messages)
            
            # Create enhanced query with conversation context
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
            
            # For document augmentation, we need to retrieve and generate answer
            docs = rag_system.get_relevant_documents(enhanced_query)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                response = f"Based on the enhanced documents, here's the answer to your query:\n\n{context}"
            else:
                response = "I couldn't find relevant information in the enhanced documents to answer your query."
        
        elif technique == "Basic RAG":
            # Build conversation context from previous messages
            conversation_context = ""
            if len(session_messages) > 1:
                conversation_context = build_conversation_context(session_messages)
            
            # Create enhanced query with conversation context
            enhanced_query = query
            if conversation_context:
                enhanced_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"
            
            # Basic similarity search
            docs = rag_system.similarity_search(enhanced_query, k=3)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                response = f"Based on the documents, here's the answer:\n\n{context}"
            else:
                response = "I couldn't find relevant information in the documents to answer your query."
        
        elif technique == "Explainable Retrieval":
            try:
                result = rag_system.answer_query(query)
                if isinstance(result, dict):
                    response = result.get('answer', str(result))
                    context = result.get('context', '')
                else:
                    response = str(result)
            except Exception as er_error:
                print(f"Explainable Retrieval error: {er_error}")
                response = f"Error with Explainable Retrieval: {str(er_error)}"
                
        return response, context, source_chunks
        
    except Exception as e:
        return f"Error generating response: {str(e)}", "", None

def build_conversation_context(messages, max_turns=3):
    """Build a context string from the last N conversation turns"""
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
                file_name = os.path.basename(file_path)
                
                # Create a document with metadata
                doc = Document(
                    page_content=content,
                    metadata={"source": file_name, "file_path": file_path}
                )
                all_documents.append(doc)
                processed_files.append(file_name)
                
            except Exception as e:
                print(f"Could not process {os.path.basename(file_path)}: {str(e)}")
        
        if not all_documents:
            raise ValueError("No documents could be processed successfully")
        
        print(f"Loaded {len(processed_files)} documents: {', '.join(processed_files)}")
        
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
        
        print(f"Created vectorstore with {len(cleaned_texts)} chunks from {len(processed_files)} documents")
        
        return vectorstore
        
    except Exception as e:
        print(f"Error creating multi-document Basic RAG: {str(e)}")
        return None

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

def get_or_create_session_id() -> str:
    """Generate a new session ID"""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"

# === FASTAPI ROUTES ===

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global evaluation_manager
    init_chat_database()
    evaluation_manager = EvaluationManager()
    print("FastAPI Multi-RAG Chatbot started successfully!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    session_id = get_or_create_session_id()
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
    
    return templates.TemplateResponse("index.html", {
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
    
    # Load RAG system
    if query_request.technique not in rag_systems:
        rag_system = load_rag_system(
            query_request.technique,
            session_data.uploaded_documents,
            query_request.crag_web_search_enabled
        )
        if rag_system:
            rag_systems[query_request.technique] = rag_system
        else:
            return JSONResponse({
                "success": False,
                "error": f"Failed to load {query_request.technique}"
            })
    else:
        rag_system = rag_systems[query_request.technique]
    
    # Generate response
    start_time = time.time()
    response, context, source_chunks = get_rag_response(
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
    
    sessions[query_request.session_id] = session_data
    
    return JSONResponse({
        "success": True,
        "response": response,
        "technique": query_request.technique,
        "query_id": assistant_message.get("query_id"),
        "source_chunks": source_chunks,
        "processing_time": response_time
    })

@app.post("/feedback")
async def submit_feedback(feedback_request: FeedbackRequest):
    """Submit user feedback"""
    try:
        feedback = UserFeedback(
            helpfulness=feedback_request.helpfulness,
            accuracy=feedback_request.accuracy,
            clarity=feedback_request.clarity,
            overall_rating=feedback_request.overall_rating,
            comments=feedback_request.comments
        )
        
        evaluation_manager.add_user_feedback(feedback_request.query_id, feedback)
        
        return JSONResponse({
            "success": True,
            "message": "Thank you for your feedback!"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@app.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    return JSONResponse(get_all_chat_sessions())

@app.post("/sessions/new")
async def create_new_session():
    """Create a new chat session"""
    session_id = get_or_create_session_id()
    session_data = get_session(session_id)
    return JSONResponse({
        "success": True,
        "session_id": session_id
    })

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    success = delete_chat_session(session_id)
    if session_id in sessions:
        del sessions[session_id]
    
    return JSONResponse({
        "success": success
    })

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
