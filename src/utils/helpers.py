from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from google.api_core.exceptions import ResourceExhausted
from typing import List
from rank_bm25 import BM25Okapi
import fitz
import asyncio
import random
import textwrap
import numpy as np
from enum import Enum
import os


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using local embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store using Gemini (GoogleGenerativeAIEmbeddings)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a string into a vector store using local embeddings.

    Args:
        content (str): The text content to be encoded.
        chunk_size (int): The size of each chunk of text.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        FAISS: A vector store containing the encoded content.

    Raises:
        ValueError: If the input content is not valid.
        RuntimeError: If there is an error during the encoding process.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata['relevance_score'] = 1.0

        # Generate embeddings and create the vector store using local embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"An error occurred during the encoding process: {str(e)}")

    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.

    Args:
        question: The question for which to retrieve context and URLs.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of unique URLs from the metadata of the relevant documents.
    """

    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]

    return context


class QuestionAnswerFromContext(BaseModel):
    """
    Model to generate an answer to a query based on a given context.
    
    Attributes:
        answer_based_on_content (str): The generated answer based on the context.
    """
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain(llm):
    # Initialize the language model with specific parameters
    question_answer_from_context_llm = llm

    # Define the prompt template for chain-of-thought reasoning
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # Create a chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_llm.with_structured_output(
        QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Answer a question using the given context by invoking a chain of reasoning.

    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.

    Returns:
        A dictionary containing the answer, context, and question.
    """
    input_data = {
        "question": question,
        "context": context
    }
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def show_context(context):
    """
    Display the contents of the provided context list.

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


def read_pdf_to_string(path):
    """
    Read a PDF document from the specified path and return its content as a string.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The concatenated text content of all pages in the PDF document.

    The function uses the 'fitz' library (PyMuPDF) to open the PDF document, iterate over each page,
    extract the text content from each page, and append it to a single string.
    """
    # Open the PDF document located at the specified path
    doc = fitz.open(path)
    content = ""
    # Iterate over each page in the document
    for page_num in range(len(doc)):
        # Get the current page
        page = doc[page_num]
        # Extract the text content from the current page and append it to the content string
        content += page.get_text()
    return content


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """
    Perform BM25 retrieval and return the top k cleaned text chunks.

    Args:
    bm25 (BM25Okapi): Pre-computed BM25 index.
    cleaned_texts (List[str]): List of cleaned text chunks corresponding to the BM25 index.
    query (str): The query string.
    k (int): The number of text chunks to retrieve.

    Returns:
    List[str]: The top k cleaned text chunks based on BM25 scores.
    """
    # Tokenize the query
    query_tokens = query.split()

    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top k scores
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Retrieve the top k cleaned text chunks
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts


async def exponential_backoff(attempt):
    """
    Implements exponential backoff with a jitter.
    
    Args:
        attempt: The current retry attempt number.
        
    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt) + a random fraction of a second.
    """
    # Calculate the wait time with exponential backoff and jitter
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")

    # Asynchronously sleep for the calculated wait time
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Retries a coroutine using exponential backoff upon encountering a ResourceExhausted error.
    
    Args:
        coroutine: The coroutine to be executed.
        max_retries: The maximum number of retry attempts.
        
    Returns:
        The result of the coroutine if successful.
        
    Raises:
        The last encountered exception if all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to execute the coroutine
            return await coroutine
        except ResourceExhausted as e:
            # If the last attempt also fails, raise the exception
            if attempt == max_retries - 1:
                raise e

            # Wait for an exponential backoff period before retrying
            await exponential_backoff(attempt)

    # If max retries are reached without success, raise an exception
    raise Exception("Max retries reached")


# Enum class representing different embedding providers
class EmbeddingProvider(Enum):
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    AMAZON_BEDROCK = "bedrock"
    GOOGLE = "google"

# Enum class representing different model providers
class ModelProvider(Enum):
    GOOGLE = "google"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AMAZON_BEDROCK = "bedrock"


def get_langchain_embedding_provider(provider: EmbeddingProvider, model_id: str = None):
    """
    Returns an embedding provider based on the specified provider and model ID.

    Args:
        provider (EmbeddingProvider): The embedding provider to use.
        model_id (str): Optional -  The specific embeddings model ID to use .

    Returns:
        A LangChain embedding provider instance.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    if provider == EmbeddingProvider.HUGGINGFACE or provider == EmbeddingProvider.GOOGLE:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model_name = model_id if model_id else "models/embedding-001"
        return GoogleGenerativeAIEmbeddings(model=model_name)
    elif provider == EmbeddingProvider.COHERE:
        try:
            from langchain_cohere import CohereEmbeddings
            return CohereEmbeddings()
        except ImportError:
            raise ValueError("Cohere embeddings not available. Install langchain-cohere to use this provider.")
    elif provider == EmbeddingProvider.AMAZON_BEDROCK:
        from langchain_community.embeddings import BedrockEmbeddings
        return BedrockEmbeddings(model_id=model_id) if model_id else BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")



def encode_document(file_paths, chunk_size=1000, chunk_overlap=200):
    """
    Encodes one or more document files into a single vectorstore using embeddings.
    Supports: PDF, TXT, CSV, JSON, DOCX, XLSX
    Args:
        file_paths: A list of document file paths (or a single path as string).
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.
    Returns:
        A FAISS vector store containing the encoded content from all documents.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    all_cleaned_texts = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        # Choose appropriate loader based on file type
        documents = None  # Initialize documents variable
        loader = None     # Initialize loader variable
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            # Try direct file reading first (more reliable for temp files)
            documents = None  # Initialize to track if we need to use loader
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create a fake document object for compatibility
                class SimpleDoc:
                    def __init__(self, content, source):
                        self.page_content = content
                        self.metadata = {'source': source}
                
                documents = [SimpleDoc(content, file_path)]
                print(f"✅ Used direct file reading for text file: {file_path}")
                
            except UnicodeDecodeError:
                # Try multiple encodings for text files if UTF-8 fails
                loader = None
                for encoding in ['utf-16', 'latin-1', 'cp1252']:
                    try:
                        loader = TextLoader(file_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                
                if loader is None:
                    # Final fallback: try direct reading with different encodings
                    for encoding in ['utf-16', 'latin-1', 'cp1252']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            
                            class SimpleDoc:
                                def __init__(self, content, source):
                                    self.page_content = content
                                    self.metadata = {'source': source}
                            
                            documents = [SimpleDoc(content, file_path)]
                            print(f"✅ Used direct file reading with {encoding} encoding: {file_path}")
                            break
                        except Exception:
                            continue
                    else:
                        raise ValueError(f"Could not decode text file {file_path} with any supported encoding")
                # If loader is not None, we'll use it below to load documents
            except Exception as e:
                raise ValueError(f"Could not read text file {file_path}: {e}")
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension == '.json':
            loader = JSONLoader(file_path, jq_schema='.', text_content=False)
        elif file_extension in ['.docx', '.doc']:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported formats: PDF, TXT, CSV, JSON, DOCX, XLSX")
        
        # Load documents (only call loader.load() if we have a loader and haven't already loaded documents)
        if documents is None and loader is not None:
            # We have a loader, use it
            try:
                documents = loader.load()
                print(f"✅ Successfully loaded using {type(loader).__name__}: {file_path}")
            except Exception as e:
                print(f"❌ Loader failed for {file_path}: {e}")
                # Try fallback loading for any file type
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    class SimpleDoc:
                        def __init__(self, content, source):
                            self.page_content = content
                            self.metadata = {'source': source}
                    
                    documents = [SimpleDoc(content, file_path)]
                    print(f"✅ Used fallback direct reading for: {file_path}")
                except Exception as fallback_error:
                    raise ValueError(f"Could not load {file_path} with any method. Loader error: {e}, Fallback error: {fallback_error}")
        elif documents is None:
            # No documents and no loader - this should not happen
            raise ValueError(f"No documents loaded and no loader available for {file_path}")
        # If documents already exists, we used direct file reading
        
        print(f"✅ Successfully loaded {file_extension.upper()} file: {os.path.basename(file_path)} ({len(documents)} document(s))")
        
        # Debug: Print original document page info
        print(f"\n=== Original PDF documents (before chunking) ===")
        for i, doc in enumerate(documents):
            page_info = doc.metadata.get('page', 'MISSING')
            print(f"Document {i}: Page: {page_info}, Content: {doc.page_content[:100]}...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        texts = text_splitter.split_documents(documents)
        cleaned_texts = replace_t_with_space(texts)
        
        # Add metadata: source (filename), page (if available), paragraph (chunk index)
        for i, doc in enumerate(cleaned_texts):
            doc.metadata['source'] = file_path
            if file_extension == '.pdf':
                # Fix page number assignment
                original_page = doc.metadata.get('page', None)
                
                # If page is missing, 0, or invalid, try to map it properly
                if original_page is None or original_page == 0:
                    # Find the source document this chunk came from by matching content
                    chunk_start = doc.page_content[:100]  # First 100 chars
                    mapped_page = 1  # Default fallback
                    
                    for j, orig_doc in enumerate(documents):
                        if chunk_start in orig_doc.page_content:
                            # Found the source document, use its page number
                            orig_page = orig_doc.metadata.get('page', j + 1)
                            mapped_page = orig_page if orig_page and orig_page > 0 else j + 1
                            break
                    
                    doc.metadata['page'] = mapped_page
                    print(f"Chunk {i}: Mapped page {mapped_page} for chunk starting with: {chunk_start[:50]}...")
                else:
                    # Use the original page number if it's valid
                    doc.metadata['page'] = original_page
                    print(f"Chunk {i}: Using original page {original_page}")
            else:
                doc.metadata['page'] = None
            doc.metadata['paragraph'] = i + 1
        all_cleaned_texts.extend(cleaned_texts)
    # Create embeddings and vector store using all chunks from all docs
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(all_cleaned_texts, embeddings)
    print(f"Successfully processed {len(all_cleaned_texts)} chunks from {len(file_paths)} file(s)")
    return vectorstore