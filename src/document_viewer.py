"""
Document Viewer Component for RAG Systems
Provides functionality to view source documents with highlighted text chunks
"""

import streamlit as st
import os
import base64
import urllib.parse
import re
import difflib
from typing import List, Dict, Any
from src.document_augmentation import load_document_content



def highlight_text_in_document(document_path: str, chunks_to_highlight: List[Dict]) -> str:
    """
    Load document and highlight specified text chunks
    
    Args:
        document_path: Path to the document file
        chunks_to_highlight: List of dictionaries with 'text' key containing chunk content
    
    Returns:
        HTML string with highlighted content
    """
    try:
        print("\n" + "="*60)
        print("🎯 DOCUMENT VIEWER CHUNK FILTERING DEBUG")
        print("="*60)
        
        # Load document content
        content = load_document_content(document_path)
        if not content:
            return f"Error: Could not load document content from {document_path}"
        
        print(f"📄 Document: {os.path.basename(document_path)}")
        print(f"📊 Content length: {len(content)} characters")
        print(f"📦 Total chunks received: {len(chunks_to_highlight)}")
        
        print("\nTest chunks:")
        for i, chunk in enumerate(chunks_to_highlight):
            score = chunk.get('score', 'N/A')
            text = chunk.get('text', '')[:50] + '...' if len(chunk.get('text', '')) > 50 else chunk.get('text', '')
            print(f"  {i+1}. Score: {score}, Text: '{text}'")
        
        print("\n=== APPLYING FILTERS ===")
        
        # Create highlighted version
        highlighted_content = content
        
        # STEP 1: Score Filter
        print(f"Step 1: Score filter (>= 0.5)")
        relevant_chunks = [
            chunk for chunk in chunks_to_highlight 
            if chunk.get('score', 0) >= 0.5  # Only highlight chunks with decent relevance
        ]
        print(f"After score filter: {len(relevant_chunks)} chunks")
        for chunk in relevant_chunks:
            score = chunk.get('score', 'N/A')
            text = chunk.get('text', '')[:50] + '...' if len(chunk.get('text', '')) > 50 else chunk.get('text', '')
            print(f"  - Score: {score}, Text: '{text}'")
        
        # STEP 2: Sort by length
        sorted_chunks = sorted(relevant_chunks, key=lambda x: len(x.get('text', '')), reverse=True)
        print(f"\nStep 2: Sorted by length (longest first)")
        for i, chunk in enumerate(sorted_chunks):
            text = chunk.get('text', '')
            print(f"  {i+1}. Length: {len(text)}, Text: '{text[:50]}...'")
        
        # STEP 3: Apply filters and highlight
        print(f"\nStep 3: Applying additional filters and highlighting")
        highlighted_count = 0
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = chunk.get('text', '').strip()
            print(f"\n--- Processing chunk {i+1} ---")
            print(f"Original text length: {len(chunk_text)}")
            print(f"Text preview: '{chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}'")
            
            # Length check
            if not chunk_text or len(chunk_text) < 20:
                print(f"❌ FILTERED OUT: Too short ({len(chunk_text)} chars < 20)")
                continue
            else:
                print(f"✅ Length check passed: {len(chunk_text)} chars >= 20")
                
            # Generic phrase check
            generic_phrases = [
                "in this document", "we will explore", "we will examine", 
                "this section discusses", "the following", "as mentioned",
                "it is important to note", "furthermore", "in addition"
            ]
            has_generic = any(phrase in chunk_text.lower() for phrase in generic_phrases)
            is_short = len(chunk_text) < 100
            
            print(f"Generic phrase check:")
            for phrase in generic_phrases:
                if phrase in chunk_text.lower():
                    print(f"  ⚠️  Found generic phrase: '{phrase}'")
            print(f"Has generic phrase: {has_generic}")
            print(f"Is short text: {is_short} ({len(chunk_text)} chars)")
            
            if has_generic and is_short:
                print(f"❌ FILTERED OUT: Generic phrase + short text")
                continue
            else:
                print(f"✅ Generic phrase check passed")
                
            # Smart text matching - find and highlight the complete section
            highlighted_found = False
            
            print(f"🔍 Smart matching for chunk:")
            print(f"   Chunk length: {len(chunk_text)} chars")
            print(f"   Preview: '{chunk_text[:150]}...'")
            
            # Strategy 1: Look for section headers/key phrases to find the right content area
            key_phrases = []
            lines = chunk_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    # Look for headers, bullet points, and substantial content
                    if (line.startswith('The ') or line.startswith('Chapter') or 
                        line.startswith('a)') or line.startswith('b)') or 
                        line.startswith('-') or ':' in line or len(line) > 30):
                        key_phrases.append(line)
            
            print(f"   Key phrases found: {len(key_phrases)}")
            for i, phrase in enumerate(key_phrases[:3]):
                print(f"     {i+1}. '{phrase}'")
            
            # Strategy 2: Find the section boundaries in the document
            if key_phrases and not highlighted_found:
                print(f"🎯 Searching for section boundaries...")
                
                # Look for the first key phrase to anchor our search
                anchor_phrase = key_phrases[0]
                if anchor_phrase in highlighted_content:
                    anchor_start = highlighted_content.find(anchor_phrase)
                    print(f"   Found anchor: '{anchor_phrase}' at position {anchor_start}")
                    
                    # Try to find where this section ends
                    section_start = anchor_start
                    section_end = anchor_start + len(anchor_phrase)
                    
                    # Extend to include related content
                    remaining_text = highlighted_content[section_end:]
                    
                    # Look for natural section boundaries
                    next_section_markers = [
                        '\nChapter ', '\nThe ', '\n\nConclusion', '\n\nReferences',
                        '\n\nBibliography', '\n\nAppendix'
                    ]
                    
                    min_end = len(remaining_text)
                    for marker in next_section_markers:
                        marker_pos = remaining_text.find(marker)
                        if marker_pos != -1 and marker_pos < min_end:
                            min_end = marker_pos
                    
                    # Take a reasonable section (up to 1000 chars or next section)
                    section_length = min(1000, min_end)
                    section_end = section_end + section_length
                    
                    # Extract the complete section
                    complete_section = highlighted_content[section_start:section_end].strip()
                    
                    if len(complete_section) > 50:  # Make sure we have substantial content
                        print(f"   Extracted section: {len(complete_section)} chars")
                        print(f"   Section preview: '{complete_section[:100]}...'")
                        
                        color = "hsl(45, 80%, 85%)"  # Consistent yellow highlight
                        highlighted_text = f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; border: 1px solid #ccc;">{complete_section}</mark>'
                        highlighted_content = highlighted_content.replace(complete_section, highlighted_text, 1)
                        highlighted_count += 1
                        highlighted_found = True
                        print(f"🎨 Applied section highlighting with color: {color}")
            
            # Strategy 3: Individual key phrase highlighting if section approach fails
            if not highlighted_found and key_phrases:
                print(f"🔍 Falling back to individual phrase highlighting...")
                phrases_highlighted = 0
                
                for phrase in key_phrases:
                    if phrase.strip() and phrase in highlighted_content:
                        print(f"   ✅ Highlighting: '{phrase[:50]}...'")
                        color = "hsl(45, 80%, 85%)"  # Consistent yellow highlight
                        highlighted_text = f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; border: 1px solid #ccc;">{phrase}</mark>'
                        highlighted_content = highlighted_content.replace(phrase, highlighted_text, 1)
                        phrases_highlighted += 1
                
                if phrases_highlighted > 0:
                    highlighted_count += 1
                    highlighted_found = True
                    print(f"🎨 Applied phrase highlighting to {phrases_highlighted} phrases")
            
            # Strategy 4: Last resort - look for any substantial matching text
            if not highlighted_found:
                print(f"🔧 Last resort: looking for any matching content...")
                
                # Split chunk into meaningful pieces
                meaningful_pieces = []
                for line in chunk_text.split('\n'):
                    line = line.strip()
                    if len(line) > 20 and not line.startswith('-') and ':' not in line[:10]:
                        meaningful_pieces.append(line)
                
                for piece in meaningful_pieces[:3]:  # Try first 3 substantial pieces
                    if piece in highlighted_content:
                        print(f"   ✅ Found piece: '{piece[:50]}...'")
                        color = "hsl(45, 80%, 85%)"  # Consistent yellow highlight
                        highlighted_text = f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px; border: 1px solid #ccc;">{piece}</mark>'
                        highlighted_content = highlighted_content.replace(piece, highlighted_text, 1)
                        highlighted_found = True
                        break
                
                if highlighted_found:
                    highlighted_count += 1
                    print(f"🎨 Applied fallback highlighting")
            
            if not highlighted_found:
                print(f"❌ Could not match any content for this chunk")
                print(f"   Chunk start: '{chunk_text[:50]}...'")
                print(f"   Document start: '{highlighted_content[:50]}...'")
        print(f"\n=== SUMMARY ===")
        print(f"📥 Total chunks received: {len(chunks_to_highlight)}")
        print(f"🔍 After score filter: {len(relevant_chunks)}")
        print(f"🎨 Final chunks highlighted: {highlighted_count}")
        print("="*60)
        
        return highlighted_content
        
    except Exception as e:
        print(f"❌ Error in highlight_text_in_document: {str(e)}")
        return f"Error loading document: {str(e)}"

def create_document_link(document_path: str, chunks_to_highlight: List[Dict], link_text: str = "View Source Document") -> None:
    """
    Create a link that opens document viewer in new tab
    
    Args:
        document_path: Path to the source document
        chunks_to_highlight: List of text chunks to highlight
        link_text: Text to display on the link button
    """
    try:
        # Encode the parameters for URL
        chunks_text = [chunk.get('text', '') for chunk in chunks_to_highlight if chunk.get('text')]
        
        if not chunks_text:
            st.warning("No text chunks available to highlight")
            return
        
        # Create unique function name to avoid conflicts
        import json
        import urllib.parse
        doc_name = os.path.basename(document_path)
        function_id = abs(hash(doc_name + str(len(chunks_text))))
        
        # Properly escape the document path and chunks for JavaScript
        escaped_path = json.dumps(document_path)  # This handles all escaping
        escaped_chunks = json.dumps(chunks_text)  # This handles all escaping
        
        # Improved JavaScript with better error handling and debugging
        js_code = f"""
        <script>
        function openDocViewer_{function_id}() {{
            try {{
                console.log('Opening document viewer...');
                const chunks = {escaped_chunks};
                const docPath = {escaped_path};
                
                // Build the URL properly
                const baseUrl = window.location.origin + window.location.pathname;
                const params = new URLSearchParams();
                params.set('page', 'document_viewer');
                params.set('doc_path', docPath);
                params.set('chunks', JSON.stringify(chunks));
                
                const url = baseUrl + '?' + params.toString();
                console.log('Opening URL:', url);
                
                // Try to open the window
                const newWindow = window.open(url, '_blank', 'width=1200,height=800,scrollbars=yes,resizable=yes,toolbar=yes,menubar=yes');
                
                if (!newWindow) {{
                    // Fallback: show alert with URL if popup was blocked
                    alert('Pop-up blocked! Please copy this URL and open it in a new tab:\\n\\n' + url);
                }} else {{
                    console.log('Window opened successfully');
                }}
            }} catch (error) {{
                console.error('Error opening document viewer:', error);
                alert('Error opening document viewer: ' + error.message);
            }}
        }}
        </script>
        """
        
        # Create the button with improved styling and hover effects
        button_html = f"""
        {js_code}
        <button onclick="openDocViewer_{function_id}()" style="
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #0066cc;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            margin: 5px;
            transition: background-color 0.2s;
        " onmouseover="this.style.backgroundColor='#0052a3'" 
           onmouseout="this.style.backgroundColor='#0066cc'"
           title="Open document viewer in new tab">{link_text}</button>
        """
        
        st.markdown(button_html, unsafe_allow_html=True)
        
        # Add a fallback option with a copyable URL
        try:
            import urllib.parse
            base_url = "http://localhost:8501"  # Default Streamlit URL
            params = urllib.parse.urlencode({
                'page': 'document_viewer',
                'doc_path': document_path,
                'chunks': json.dumps(chunks_text)
            })
            fallback_url = f"{base_url}?{params}"
            
            # Show the URL in a small expander for manual copying if needed
            with st.expander("🔗 Manual link (if button doesn't work)", expanded=False):
                st.text_input(
                    "Copy this URL to open in new tab:",
                    value=fallback_url,
                    key=f"url_fallback_{function_id}",
                    help="If the button above doesn't work, copy this URL and paste it in a new browser tab"
                )
        except Exception as fallback_error:
            st.caption("Fallback URL generation failed")
        
    except Exception as e:
        st.error(f"Error creating document link: {str(e)}")
        # Show the error details for debugging
        st.write(f"Debug info: document_path={document_path}, chunks_count={len(chunks_to_highlight)}")

def show_embedded_document_viewer(doc_path: str, chunks_to_highlight: List[Dict], use_expander: bool = True, message_id: str = None) -> None:
    """
    Show document viewer within the main app as an expandable section
    
    Args:
        doc_path: Path to the document
        chunks_to_highlight: List of text chunks to highlight
        use_expander: Whether to wrap content in an expander (False to avoid nesting issues)
        message_id: Unique message ID for creating unique keys
    """
    doc_name = os.path.basename(doc_path)
    
    def render_content():
        # Add download button if file exists
        try:
            if os.path.exists(doc_path):
                with open(doc_path, 'rb') as file:
                    # Create unique download key using timestamp and hash to avoid collisions
                    import time
                    download_key = f"download_{message_id}_{hash(doc_path)}_{int(time.time() * 1000)}" if message_id else f"download_{hash(doc_path)}_{int(time.time() * 1000)}"
                    st.download_button(
                        label="📥 Download Original Document",
                        data=file.read(),
                        file_name=doc_name,
                        mime="application/pdf" if doc_path.endswith('.pdf') else "text/plain",
                        key=download_key
                    )
        except Exception as e:
            st.warning(f"Could not create download button: {str(e)}")
        
        # Show highlighted content
        highlighted_content = highlight_text_in_document(doc_path, chunks_to_highlight)
        
        # Create scrollable container with highlighted content
        st.markdown(f"""
        <div style="
            height: 400px; 
            overflow-y: scroll; 
            background-color: #ffffff; 
            color: #000000;
            padding: 15px; 
            border-radius: 8px;
            border: 1px solid #dee2e6;
            font-family: 'Georgia', serif;
            line-height: 1.6;
            white-space: pre-wrap;
        ">
            {highlighted_content}
        </div>
        """, unsafe_allow_html=True)
        
        # Show chunk details
        st.markdown("**📋 Source Chunks Used:**")
        for i, chunk in enumerate(chunks_to_highlight):
            chunk_text = chunk.get('text', '')
            if chunk_text:
                # Use container instead of expander to avoid nesting
                with st.container():
                    st.markdown(f"**Chunk {i+1}** (Score: {chunk.get('score', 'N/A')})")
                    st.text_area("", value=chunk_text, height=100, key=f"chunk_{i}_{hash(chunk_text)}", disabled=True)
                    if 'source' in chunk:
                        st.caption(f"Source: {chunk['source']}")
                    if 'page' in chunk:
                        st.caption(f"Page: {chunk['page']}")
                    st.divider()
    
    if use_expander:
        with st.expander(f"📄 View Source: {doc_name}", expanded=False):
            render_content()
    else:
        st.markdown(f"### 📄 View Source: {doc_name}")
        render_content()

def document_viewer_page():
    """
    Standalone document viewer page for opening in new tab
    """
    st.set_page_config(page_title="Document Viewer", layout="wide")
    st.title("📄 Source Document Viewer")
    
    # Get parameters from URL
    query_params = st.query_params
    
    if 'doc_path' in query_params and 'chunks' in query_params:
        doc_path = query_params['doc_path']
        
        try:
            # Parse chunks from JSON
            import json
            chunks_json = query_params['chunks']
            chunks_text = json.loads(chunks_json)
            chunks_to_highlight = [{'text': chunk} for chunk in chunks_text if chunk]
            
        except Exception as e:
            st.error(f"Error parsing chunks: {str(e)}")
            return
        
        doc_name = os.path.basename(doc_path)
        st.subheader(f"Document: {doc_name}")
        
        # Navigation help
        st.info("💡 Use Ctrl+F to search for specific text within this document")
        
        # Show highlighted document
        highlighted_content = highlight_text_in_document(doc_path, chunks_to_highlight)
        
        # Display with custom CSS for better formatting
        st.markdown("""
        <style>
        .document-content {
            background-color: #ffffff;
            color: #000000;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            font-family: 'Georgia', serif;
            line-height: 1.6;
            white-space: pre-wrap;
            max-height: 70vh;
            overflow-y: auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="document-content">{highlighted_content}</div>', 
                   unsafe_allow_html=True)
        
        # Show chunk information in sidebar
        with st.sidebar:
            st.subheader("📋 Highlighted Chunks")
            st.write(f"**Total chunks:** {len(chunks_to_highlight)}")
            
            for i, chunk in enumerate(chunks_to_highlight):
                with st.expander(f"Chunk {i+1}"):
                    chunk_text = chunk.get('text', '')
                    if len(chunk_text) > 200:
                        st.text(chunk_text[:200] + "...")
                    else:
                        st.text(chunk_text)
                    
                    # Show chunk metadata if available
                    if 'source' in chunk:
                        st.caption(f"📁 Source: {chunk['source']}")
                    if 'page' in chunk:
                        st.caption(f"📄 Page: {chunk['page']}")
                    if 'score' in chunk:
                        st.caption(f"🎯 Relevance: {chunk['score']}")
    else:
        st.error("❌ No document specified for viewing.")
        st.write("This page requires document path and chunks parameters.")

def check_document_viewer_page():
    """
    Check if current page is document viewer and handle accordingly
    """
    query_params = st.query_params
    if query_params.get('page') == 'document_viewer':
        document_viewer_page()
        return True
    return False
