import re

# FIXED: Changed function to accept only positional arguments
def evaluate_documents(query, documents, llm_arg=None):
    """Evaluate a list of documents against a query and return relevance scores.
    
    Args:
        query (str): The user query
        documents (list): List of document texts to evaluate
        llm_arg: Optional LLM to use (if not provided, will use the imported default)
        
    Returns:
        list: List of relevance scores (0.0-1.0) for each document
    """
    print(f"⏳ Evaluating {len(documents)} document chunks for relevance...")
    # If llm is not passed, create a default one
    evaluation_llm = llm_arg
    if evaluation_llm is None:
        from src.llm.gemini import SimpleGeminiLLM
        evaluation_llm = SimpleGeminiLLM(model="gemini-1.5-flash")
    
    scores = [retrieval_evaluator(query, doc, evaluation_llm) for doc in documents]
    print(f"✅ Document evaluation complete. Scores: {scores}")
    return scores

def retrieval_evaluator(query, document, llm):
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
    result = llm.invoke(prompt_text)
    # Extract the numeric score from the response
    try:
        score_text = result.content.strip()
        # Try to extract a number from the response
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

def evaluate_docs_v2(query, documents, llm_obj=None):
    """Alternative version of evaluate_documents to avoid caching issues.
    
    Args:
        query (str): The user query
        documents (list): List of document texts to evaluate
        llm_obj: Optional LLM to use (if not provided, will use the imported default)
        
    Returns:
        list: List of relevance scores (0.0-1.0) for each document
    """
    print(f"⏳ Evaluating {len(documents)} document chunks for relevance (v2)...")
    # If llm is not passed, create a default one
    evaluation_llm = llm_obj
    if evaluation_llm is None:
        from src.llm.gemini import SimpleGeminiLLM
        evaluation_llm = SimpleGeminiLLM(model="gemini-1.5-flash")
    
    scores = [retrieval_evaluator(query, doc, evaluation_llm) for doc in documents]
    print(f"✅ Document evaluation complete. Scores: {scores}")
    return scores