def knowledge_refinement(document, llm):
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
    print("⏳ Refining knowledge from document chunk...")
    result = llm.invoke(prompt_text)
    print("✅ Knowledge refinement complete")
    try:
        content = result.content.strip()
        points = [line.strip() for line in content.split('\n') if line.strip()]
        return points
    except AttributeError:
        return ["Error processing document"]

# def generate_response(query, knowledge, sources, llm, source_info):
#     sources_text = "\n".join([f"- {title}: {link}" if link else f"- {title}" for title, link in sources])
#     prompt_text = f"""Answer the question using ONLY the information provided in the knowledge base.
# DO NOT add information from your general knowledge or external sources.

# STRICT GUIDELINES:
# - Use ONLY the provided knowledge - nothing more
# - Include ALL relevant information from the sources that answers the question
# - For direct questions (like "how long", "what is", "when", "where", "how many", etc.), ALWAYS extract the specific answer from the text, even if it's embedded within a larger sentence or paragraph
# - When the answer is part of a larger sentence, extract and provide the direct answer to the specific question asked
# - If the answer can be directly inferred from the provided knowledge, state it clearly and concisely
# - If the answer is present in the knowledge but not as a direct sentence, synthesize a direct answer using only the provided information
# - If the knowledge is limited, be honest about limitations but provide what is available
# - Do NOT elaborate beyond the source material
# - Present information completely - don't omit parts of statements
# - Keep responses focused but comprehensive based on available sources
# - Do NOT invent details or explanations not in the sources
# - IMPORTANT: Only say you cannot answer if the information is truly not present or cannot be reasonably inferred from the knowledge

# CRITICAL: For any information from uploaded documents, include source reference: [Source: DOCUMENT_NAME, page X, paragraph Y]

# Query: {query}

# {source_info}
# Available Knowledge (use ONLY this information):
# {knowledge}

# Sources:
# {sources_text}

# Provide a complete answer using ALL the relevant information provided:

# Answer:"""
#     print("⏳ Generating final response with LLM...")
#     result = llm.invoke(prompt_text)
#     print("✅ Final response generated")
#     return result.content

def generate_response(query, knowledge, sources, llm, _last_source_chunks=None, web_search_enabled=False):
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

    if not web_search_enabled:
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

    print("⏳ Generating final response with LLM...")
    result = llm.invoke(prompt_text)
    print("✅ Final response generated")
    
    # Validate response against sources to prevent hallucination
    response_content = result.content
    
    # Basic validation: check response length vs source material
    if _last_source_chunks:
        source_text = " ".join([chunk.get('text', '') for chunk in _last_source_chunks])
        source_words = len(source_text.split())
        response_words = len(response_content.split())
        
        if response_words > source_words * 0.5:
            print(f"⚠️ Warning: Response ({response_words} words) may exceed source material ({source_words} words)")
            print("   Response may contain hallucinated information")
    
    # Format source references with color for better readability
    formatted_response = _format_sources_with_color(response_content)
    
    return formatted_response


def _format_sources_with_color(response_text):
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
