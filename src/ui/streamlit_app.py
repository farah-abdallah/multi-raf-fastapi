import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
from src.crag import CRAG

def streamlit_crag_app():
    st.title("CRAG Chatbot Demo")
    st.write("Upload one or more documents and ask questions. Optionally enable/disable web search.")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, CSV, DOCX, XLSX, JSON)", 
        accept_multiple_files=True
    )
    web_search_enabled = st.checkbox("Enable Web Search", value=True)
    use_api_rotation = st.checkbox("Enable API Key Rotation (for multiple API keys)", value=True)
    query = st.text_input("Enter your question:")

    model = st.selectbox("Model", ["gemini-1.5-flash"], index=0)
    max_tokens = st.number_input("Max tokens", min_value=100, max_value=4096, value=1000)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0)

    if uploaded_files and query:
        import tempfile, os
        tmp_paths = []
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_paths.append(tmp_file.name)
        st.info(f"Processing {len(tmp_paths)} document(s): " + ", ".join([os.path.basename(p) for p in tmp_paths]))
        # Initialize LLM with API rotation if enabled
        if use_api_rotation:
            from src.llm.api_manager import get_api_manager
            from src.llm.gemini import SimpleGeminiLLM
            api_manager = get_api_manager()
            llm = SimpleGeminiLLM(model=model, max_tokens=max_tokens, temperature=temperature, 
                                 use_rotation=True, api_manager=api_manager)
            st.info(f"Using API rotation with {len(api_manager.api_keys)} available keys")
        else:
            llm = None
        
        crag = CRAG(
            file_path=tmp_paths,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            web_search_enabled=web_search_enabled,
            llm=llm
        )
        with st.spinner("Generating answer..."):
            answer = crag.run(query)
        st.success("Answer:")
        st.write(answer)


if __name__ == "__main__":
    streamlit_crag_app()